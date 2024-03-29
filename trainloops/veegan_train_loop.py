"""
    The VEEGAN train loop
    By using the "extended reproduction step", the algorithm will not only minimize d(z, Gz(Gx(z)),
        but also d(Gz(x), Gz(Gx(Gz(x))) wrt Gx.
"""

import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop


class VEEGANTrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1, d_img_noise_std=0.0, decrease_noise=True, pre_training_steps=0, extended_reproduction_step=False):
        super().__init__(listeners, epochs)
        self.pre_training_steps = pre_training_steps
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
        self.pre_training_done = False
        self.extended_reproduction_step = extended_reproduction_step
        self.decrease_noise = decrease_noise

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()
        if not self.pre_training_done:
            self.pre_train()
            self.pre_training_done = True

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
                noise_factor = self.d_img_noise_std * \
                               (1 if not self.decrease_noise else 1 - (self.current_epoch/self.epochs))
                xr_d_inp += torch.randn_like(xr_d_inp) * noise_factor

            dis_q = self.D((xr_d_inp, z_hat.detach()))
            d_real_labels = torch.zeros_like(dis_q)
            L_d_real = F.binary_cross_entropy_with_logits(dis_q, d_real_labels, reduction="mean")
            L_d_real.backward()

            x_tilde = self.Gx(z)
            xf_d_inp = x_tilde.detach()

            # Add noise to the inputs of D if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                noise_factor = self.d_img_noise_std * \
                               (1 if not self.decrease_noise else 1 - (self.current_epoch/self.epochs))
                xf_d_inp += torch.randn_like(xf_d_inp) * noise_factor

            dis_p = self.D((xf_d_inp, z))
            L_d_fake = F.binary_cross_entropy_with_logits(dis_p, torch.ones_like(dis_q), reduction="mean")
            L_d_fake.backward()

            L_d = L_d_real + L_d_fake

            # Gradient update on Discriminator network
            # torch.nn.utils.clip_grad_norm_(list(self.D.parameters()), 1.0)
            if L_d.detach().item() > 0.001:
                self.optim_D.step()
            else:
                self.optim_D.zero_grad()

            # Train G
            self.optim_G.zero_grad()

            if self.extended_reproduction_step:
                gz_x = self.Gz(x)[0].detach()
                loss_g_extended = self.l2_loss(self.Gz(self.Gx(gz_x))[0], gz_x)
                loss_g_extended.backward()

                # Remove gradients for Gz
                self.Gz.zero_grad()

            # Compute and backpropagate loss for x_tilde
            z = self.generate_z_batch(self.batch_size)
            x_tilde = self.Gx(z)
            dis_p = self.D((x_tilde, z))
            L_gx = F.binary_cross_entropy_with_logits(dis_p, torch.zeros_like(dis_q), reduction="mean")
            L_gx.backward(retain_graph=True)

            # Reconstruct z and backpropagate the z reconstruction loss
            z_recon, _, _ = self.Gz(x_tilde)
            z_recon_loss = self.l2_loss(z_recon, z)
            z_recon_loss.backward()

            loss_g = z_recon_loss + L_gx
            # loss_g.backward()

            # torch.nn.utils.clip_grad_norm_(list(self.Gx.parameters()) + list(self.Gz.parameters()), 1.0)
            self.optim_G.step()

        losses = {
                "D_loss": L_d.detach().item(),
                "Gx_gan_loss": L_gx.detach().item(),
                "z_reconstruction_loss": z_recon_loss.detach().item()
            }

        if self.extended_reproduction_step:
            losses["Extended_G_loss"] = loss_g_extended.detach().item()

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

    def pre_train(self):
        for i in range(self.pre_training_steps):
            self.optim_G.zero_grad()
            z = self.generate_z_batch(self.batch_size)
            x_tilde = self.Gx(z).detach()
            z_recon, _, _ = self.Gz(x_tilde)
            z_recon_loss = self.l2_loss(z_recon, z, False)
            z_recon_loss.backward()
            self.optim_G.step()



    @staticmethod
    def l2_loss(pred, target, mean=False):
        N = pred.size(0)
        loss = torch.pow(pred - target, 2).sum()/N
        if mean:
            latent_size = pred.size(1)
            loss /= latent_size
        return loss

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
