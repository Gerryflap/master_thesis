"""
    An ALI/MorGAN trainloop that splits the latent vector into two parts:
    a part that should be (nearly) equal between different images of the same person and a part
    where this requirement is not present.
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop


def get_log_odds(raw_marginals, use_sigmoid):
    if use_sigmoid:
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    else:
        # Correct for normalization between -1 and 1
        raw_marginals = (raw_marginals + 1)/2
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


class SplitMorGANTrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1,
                 morgan_alpha=0.0, d_img_noise_std=0.0, d_real_label=1.0, decrease_noise=True, use_sigmoid=True,
                 reconstruction_loss_mode="pixelwise", constrained_latent_size=128, constrained_latent_loss_factor=1.0):
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
        self.constrained_latent_size = constrained_latent_size
        if self.constrained_latent_size > self.Gx.latent_size:
            raise ValueError("The constrained latent size cannot be larger than the latent vector!")

        if reconstruction_loss_mode not in ["pixelwise", "dis_l"]:
            raise ValueError("Reconstruction loss mode must be one of \"pixelwise\" or \"dis_l\"")
        self.reconstruction_loss_mode = reconstruction_loss_mode
        self.cll_factor = constrained_latent_loss_factor

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        for i, (x, x2, _) in enumerate(self.dataloader):
            if x.size()[0] != self.batch_size:
                continue

            # Train D
            # Draw M (= batch_size) *2 samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x = x.cuda()
                x2 = x2.cuda()

            if self.current_epoch == 0 and i == 0:
                    if hasattr(self.Gx, 'output_bias'):
                        self.Gx.output_bias.data = get_log_odds(x, self.use_sigmoid)
                    else:
                        print("WARNING! Gx does not have an \"output_bias\". "
                              "Using untied biases as the last layer of Gx is advised!")

            # ========== Computations for Dis(x, z_hat) ==========

            x_no_noise = x
            # Add noise to the inputs if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                x = self.add_instance_noise(x)

            # Sample from conditionals (sampling is implemented by models)
            z_hat, z_mean, _ = self.Gz(x)
            dis_q = self.D((x, z_hat))

            z2_hat, z2_mean, _ = self.Gz(x2)

            # Compute L_latent
            L_latent = torch.nn.functional.mse_loss(
                z_mean[:, :self.constrained_latent_size],
                z2_mean[:, :self.constrained_latent_size]
            )

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

            if self.morgan:
                x_recon = self.Gx(z_hat)
                if self.reconstruction_loss_mode == "pixelwise":
                    L_pixel = self.morgan_pixel_loss(x_recon, x_no_noise)
                else:
                    L_pixel = self.dis_l_loss(x_recon, x_no_noise)
                L_syn = L_g + self.morgan_alpha * L_pixel
            else:
                L_syn = L_g

            L_syn += self.cll_factor * L_latent
            # ========== Back propagation and updates ==========

            # Gradient update on Discriminator network
            if L_g.detach().item() < 3.5:
                self.optim_D.zero_grad()
                L_d.backward(retain_graph=True)
                self.optim_D.step()

            # Gradient update on the Generator networks
            self.optim_G.zero_grad()

            L_syn.backward()

            self.optim_G.step()

        losses = {
                "D_loss": L_d.detach().item(),
                "G_loss": L_g.detach().item(),
                "L_latent": L_latent.detach().item()
            }
        if self.morgan:
            losses["L_pixel"] = L_pixel.detach().item()
            losses["L_syn"] = L_syn.detach().item()


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

    @staticmethod
    def morgan_pixel_loss(x_recon, target):
        absolute_errors = torch.abs(x_recon - target)
        # WxH = float(int(absolute_errors.size()[2]) * int(absolute_errors.size()[3]))
        # loss = absolute_errors.sum()/WxH
        loss = absolute_errors.mean()
        return loss

    def dis_l_loss(self, prediction, target):
        _, dis_l_prediction = self.D.compute_dx(prediction)
        _, dis_l_target = self.D.compute_dx(target)
        return torch.nn.functional.mse_loss(dis_l_prediction, dis_l_target)

    def add_instance_noise(self, x):
        noise_factor = self.d_img_noise_std * \
                       (1 if not self.decrease_noise else 1 - (self.current_epoch / self.epochs))
        return x + torch.randn_like(x) * noise_factor
