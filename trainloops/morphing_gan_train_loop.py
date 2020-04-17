"""
    This trainloop is an extension to ALI (and MorGAN)
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop
from util.interpolation import torch_slerp
from util.torch.losses import euclidean_distance


def get_log_odds(raw_marginals, use_sigmoid):
    if use_sigmoid:
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    else:
        # Correct for normalization between -1 and 1
        raw_marginals = (raw_marginals + 1)/2
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


class MorphingGANTrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1, morgan_alpha=0.0,
                 d_img_noise_std=0.0, d_real_label=1.0, decrease_noise=True, use_sigmoid=True,
                 morph_loss_factor=0.0, reconstruction_loss_mode="pixelwise", morph_loss_mode="pixelwise",
                 frs_model=None, unlock_D=False, random_interpolation=False, slerp=False, no_morph_loss_on_Gz=False,
                 no_morph_loss_on_Gx=False, trainable_morph_network_consistency_loss=False, max_morph_loss=False):
        super().__init__(listeners, epochs)
        self.use_sigmoid = use_sigmoid
        self.batch_size = dataloader.batch_size
        self.Gz = Gz
        self.Gx = Gx
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.dataloader = dataloader
        self.cuda = cuda
        self.morgan_alpha = morgan_alpha
        self.morgan = morgan_alpha != 0
        self.d_img_noise_std = d_img_noise_std
        self.d_real_label = d_real_label
        self.decrease_noise = decrease_noise
        self.frs_model = frs_model
        self.unlock_D = unlock_D
        self.no_morph_loss_on_Gz = no_morph_loss_on_Gz
        self.no_morph_loss_on_Gx = no_morph_loss_on_Gx
        self.trainable_morph_network_consistency_loss = trainable_morph_network_consistency_loss
        self.max_morph_loss = max_morph_loss


        # Sample morph z's along the entire line between z1 and z2 randomly instead of only in the middle
        self.random_interpolation = random_interpolation

        # Use slerp interpolation instead of linear
        self.slerp = slerp

        if reconstruction_loss_mode not in ["pixelwise", "dis_l", "frs"]:
            raise ValueError("Reconstruction loss mode must be one of \"pixelwise\", \"dis_l\" or \"frs\"")
        self.reconstruction_loss_mode = reconstruction_loss_mode

        if morph_loss_mode not in ["pixelwise", "dis_l", "frs"]:
            raise ValueError("Reconstruction loss mode must be one of \"pixelwise\", \"dis_l\" or \"frs\"")
        self.morph_loss_mode = morph_loss_mode
        self.morph_loss_factor = morph_loss_factor

        if max_morph_loss and random_interpolation:
            print("WARNING! Using max morph loss with random interpolation will probably result in weird behaviour! "
                  "The max morph loss is not scaled with beta and as such will always be minimal around 50%/50%.")
        if max_morph_loss and morph_loss_mode == "frs":
            print("Note: using max morph loss is always enabled for the frs based loss. Enabling the flag has no effect"
                  )

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        for i, (x1, x2) in enumerate(self.dataloader):
            if x1.size(0) != self.batch_size:
                continue

            # Train D
            # Draw M (= batch_size) samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
            x_merged = torch.cat([x1, x2], dim=0)

            if self.current_epoch == 0 and i == 0:
                    if hasattr(self.Gx, 'output_bias'):
                        self.Gx.output_bias.data = get_log_odds(x_merged, self.use_sigmoid)
                    else:
                        print("WARNING! Gx does not have an \"output_bias\". "
                              "Using untied biases as the last layer of Gx is advised!")

            # ========== Computations for Dis(x, z_hat) ==========

            # Add noise to the inputs if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                x = self.add_instance_noise(x1)
            else:
                x = x1

            # Sample from conditionals (sampling is implemented by models)
            z1_hat = self.Gz.encode(x)
            dis_q = self.D((x, z1_hat))

            # ========== Computations for Dis(x_tilde, z) ==========

            z = self.generate_z_batch(self.batch_size)
            x_tilde = self.Gx(z)
            # Add noise to the inputs of D if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                x_tilde = self.add_instance_noise(x_tilde)

            dis_p = self.D((x_tilde, z))

            # ========== Loss computations ==========

            L_d_fake = F.binary_cross_entropy_with_logits(dis_p, torch.zeros_like(dis_q))
            d_real_labels = torch.ones_like(dis_q) * self.d_real_label
            L_d_real = F.binary_cross_entropy_with_logits(dis_q, d_real_labels)
            L_d = L_d_real + L_d_fake

            L_g_fake = F.binary_cross_entropy_with_logits(dis_p, torch.ones_like(dis_q))
            L_g_real = F.binary_cross_entropy_with_logits(dis_q, torch.zeros_like(dis_q))

            L_g = L_g_real + L_g_fake

            x_recon = self.Gx(z1_hat)

            z2_hat = self.Gz.encode(x2)

            if not self.random_interpolation:
                beta = 0.5
            else:
                # Sample random interpolation points for every z in the batch
                beta = torch.rand((self.batch_size, 1))
                if self.cuda:
                    beta = beta.cuda()

            if not self.slerp:
                if beta == 0.5:
                    z_morph = self.Gz.morph_zs(z1_hat, z2_hat)
                else:
                    z_morph = beta * z1_hat + (1 - beta) * z2_hat
            else:
                z_morph = torch_slerp(beta, z2_hat, z1_hat, dim=1)

            if self.no_morph_loss_on_Gz:
                z_morph = z_morph.detach()
            x_morph = self.Gx(z_morph)
            if self.reconstruction_loss_mode == "pixelwise":
                dis_l_x1 = None
                L_recon = self.reconstruction_loss(x_recon, x1)
            elif self.reconstruction_loss_mode == "dis_l":
                _, dis_l_recon = self.D.compute_dx(x_recon)
                _, dis_l_x1 = self.D.compute_dx(x1)
                L_recon = self.reconstruction_loss(dis_l_recon, dis_l_x1)
            else:
                dis_l_x1 = None
                L_recon = euclidean_distance(self.frs_model(x_recon), self.frs_model(x1))

            if self.morph_loss_factor != 0.0:
                if self.morph_loss_mode == "pixelwise":
                    L_morph = self.morph_loss(x_morph, x1, x2, beta)
                elif self.morph_loss_mode == "dis_l":
                    _, dis_l_morph = self.D.compute_dx(x_morph)
                    _, dis_l_x2 = self.D.compute_dx(x2)
                    if dis_l_x1 is None:
                        _, dis_l_x1 = self.D.compute_dx(x1)
                    L_morph = self.morph_loss(dis_l_morph, dis_l_x1, dis_l_x2, beta)
                else:
                    dist1 = euclidean_distance(self.frs_model(x_morph), self.frs_model(x1))
                    dist2 = euclidean_distance(self.frs_model(x_morph), self.frs_model(x2))
                    L_morph = torch.max(dist1, dist2)
            else:
                # Do not compute L_morph if it is no needed
                L_morph = torch.zeros((1,), dtype=torch.float32)
                if self.cuda:
                    L_morph = L_morph.cuda()
            L_syn_without_morph_loss = L_g + self.morgan_alpha * L_recon
            L_syn = L_syn_without_morph_loss + self.morph_loss_factor * L_morph

            if self.trainable_morph_network_consistency_loss:
                z_in = self.generate_z_batch(self.batch_size)
                z_out = self.Gz.morph_zs(z_in, z_in)
                morph_cons_loss = torch.nn.functional.mse_loss(z_out, z_in)

                # Added 9 April 2020
                z_morph_detached_from_Gz = self.Gz.morph_zs(z1_hat.detach(), z2_hat.detach())
                l2_norms = torch.max(z1_hat.detach().norm(2, dim=1), z1_hat.detach().norm(2, dim=1))
                morph_l2_norms = z_morph_detached_from_Gz.norm(2, dim=1)
                morph_scale_loss = (torch.nn.functional.relu(morph_l2_norms - l2_norms)**2).mean()
                morph_cons_loss += 10.0 * morph_scale_loss

            # ========== Back propagation and updates ==========

            # Gradient update on Discriminator network
            if L_g.detach().item() < 3.5 or self.unlock_D:
                self.optim_D.zero_grad()
                L_d.backward(retain_graph=True)
                self.optim_D.step()

            # Gradient update on the Generator networks
            self.optim_G.zero_grad()

            if self.morph_loss_factor != 0.0:
                L_morph_weighted = L_morph * self.morph_loss_factor
                L_morph_weighted.backward(retain_graph=True)
                if self.no_morph_loss_on_Gx:
                    self.Gx.zero_grad()

            L_syn_without_morph_loss.backward()

            if self.trainable_morph_network_consistency_loss:
                morph_cons_loss.backward()

            self.optim_G.step()

        losses = {
                "D_loss": L_d.detach().item(),
                "G_loss": L_g.detach().item(),
                "L_recon": L_recon.detach().item(),
                "L_morph": L_morph.detach().item(),
                "L_syn": L_syn.detach().item()
            }
        if self.trainable_morph_network_consistency_loss:
            losses["Morph consistency loss"] = morph_cons_loss.detach().item()

        return {
            "epoch": self.current_epoch,
            "losses": losses,
            "networks": {
                "Gx": self.Gx,
                "Gz": self.Gz,
                "D": self.D,
            },
            "optimizers": {
                "G_optimizer": self.optim_G,
                "D_optimizer": self.optim_D
            }
        }

    def generate_z_batch(self, batch_size):
        z = torch.normal(torch.zeros((batch_size, self.Gx.latent_size)), 1)
        if self.cuda:
            z = z.cuda()
        return z

    def generate_batch(self, batch_size):
        # Generate random latent vectors
        z = self.generate_z_batch(batch_size)

        # Return outputs
        return self.Gx(z)

    def reconstruction_loss(self, x_recon, target):
        if self.reconstruction_loss_mode == "pixelwise":
            return self.morgan_pixel_loss(x_recon, target).mean()
        else:
            return self.dis_l_loss(x_recon, target).mean()

    def morph_loss(self, x_morph, x1, x2, beta):
        if self.morph_loss_mode == "pixelwise":
            x1_loss = self.morgan_pixel_loss(x_morph, x1)
            x2_loss = self.morgan_pixel_loss(x_morph, x2)

        else:
            x1_loss = self.dis_l_loss(x_morph, x1)
            x2_loss = self.dis_l_loss(x_morph, x2)

        if self.max_morph_loss:
            return torch.max(x1_loss, x2_loss).mean()
        else:
            return (beta * x1_loss + (1 - beta) * x2_loss).mean()

    def dis_l_loss(self, dis_l_prediction, dis_l_target):
        return torch.nn.functional.mse_loss(dis_l_prediction, dis_l_target, reduction='none').view(self.batch_size, -1).mean(dim=1, keepdim=True)

    @staticmethod
    def morgan_pixel_loss(x_recon, target):
        absolute_errors = torch.abs(x_recon - target)
        # WxH = float(int(absolute_errors.size()[2]) * int(absolute_errors.size()[3]))
        # loss = absolute_errors.sum()/WxH
        loss = absolute_errors.view(x_recon.size(0), -1).mean(dim=1, keepdim=True)
        return loss

    def add_instance_noise(self, x):
        noise_factor = self.d_img_noise_std * \
                       (1 if not self.decrease_noise else 1 - (self.current_epoch / self.epochs))
        return x + torch.randn_like(x) * noise_factor
