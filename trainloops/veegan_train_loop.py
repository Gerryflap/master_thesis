

"""
    Models a train loop for ALI: Adversarially Learned Inference (https://arxiv.org/abs/1606.00704)
    Additionally, this train loop can also perform the MorGAN algorithm by setting the MorGAN alpha
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop


class VEEGANTrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1, d_img_noise_std=0.0, d_real_label=1.0):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.Gz = Gz
        self.Gx = Gx
        # self.G = torch.nn.ModuleList([self.Gx, self.Gz])
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.dataloader = dataloader
        self.cuda = cuda
        self.d_img_noise_std = d_img_noise_std
        self.d_real_label = d_real_label

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        for i, (x, _) in enumerate(self.dataloader):
            if x.size()[0] != self.batch_size:
                continue

            # Train D
            # Draw M (= batch_size) samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x = x.cuda()
            z = self.generate_z_batch(self.batch_size)

            self.optim_D.zero_grad()

            # Sample from conditionals (sampling is implemented by models)
            z_hat = self.Gz.encode(x)
            xr_d_inp = x

            # Add noise to the inputs of D if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                xr_d_inp += torch.randn_like(xr_d_inp)*self.d_img_noise_std

            dis_q = self.D((xr_d_inp, z_hat.detach()))
            d_real_labels = torch.ones_like(dis_q)
            if self.d_real_label != 1.0:
                d_real_labels *= self.d_real_label
            L_d_real = F.binary_cross_entropy_with_logits(dis_q, d_real_labels, reduction="mean")
            L_d_real.backward()

            x_tilde = self.Gx(z)
            xf_d_inp = x_tilde.detach()

            # Add noise to the inputs of D if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                xf_d_inp += torch.randn_like(xf_d_inp)*self.d_img_noise_std

            dis_p = self.D((xf_d_inp, z))
            L_d_fake = F.binary_cross_entropy_with_logits(dis_p, torch.zeros_like(dis_q), reduction="mean")
            L_d_fake.backward()

            L_d = L_d_real + L_d_fake

            # Gradient update on Discriminator network
            torch.nn.utils.clip_grad_norm_(list(self.D.parameters()), 1.0)
            self.optim_D.step()

            # Train G
            self.optim_G.zero_grad()

            # Compute and backpropagate loss for x_tilde
            z = self.generate_z_batch(self.batch_size)
            x_tilde = self.Gx(z)
            dis_p = self.D((x_tilde, z))
            L_gx = F.binary_cross_entropy_with_logits(dis_p, torch.ones_like(dis_q), reduction="mean")
            # L_gx.backward(retain_graph=True)

            # Reconstruct z and backpropagate the z reconstruction loss
            z_recon, _, _ = self.Gz(x_tilde)
            z_recon_loss = torch.nn.functional.mse_loss(z_recon, z, reduction="mean")
            # z_recon_loss.backward()

            loss_g = z_recon_loss + L_gx
            loss_g.backward()

            torch.nn.utils.clip_grad_norm_(list(self.Gx.parameters()) + list(self.Gz.parameters()), 1.0)
            self.optim_G.step()
        self.Gx.eval()
        self.Gz.eval()
        self.D.eval()

        losses = {
                "D_loss": L_d.detach().item(),
                "Gx_gan_loss": L_gx.detach().item(),
                "z_reconstruction_loss": z_recon_loss.detach().item()
            }

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
