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


            # Sample from conditionals (sampling is implemented by models)
            z_hat = self.Gz.encode(x)

            # Add noise to the inputs if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                x = self.add_instance_noise(x)


            # The encoder should not get GAN gradients in VEEGAN
            dis_q = self.D((x, z_hat.detach()))
            d_real_labels = torch.ones_like(dis_q)

            x_tilde = self.Gx(z)
            x_tilde_no_noise = x_tilde

            # Add noise to the inputs of D if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                x_tilde = self.add_instance_noise(x_tilde)


            dis_p = self.D((x_tilde, z))

            L_d_fake = F.binary_cross_entropy_with_logits(dis_p, torch.zeros_like(dis_q))
            L_d_real = F.binary_cross_entropy_with_logits(dis_q, d_real_labels)
            L_d = L_d_real + L_d_fake

            L_gx_gan = F.binary_cross_entropy_with_logits(dis_p, torch.ones_like(dis_q))
            L_gz_gan = F.binary_cross_entropy_with_logits(dis_q, torch.zeros_like(dis_q))
            z_recon, _, _ = self.Gz(x_tilde_no_noise)
            z_recon_loss = self.l2_loss(z_recon, z, mean=True)
            L_gan = L_gx_gan + L_gz_gan

            L_g = L_gx_gan + z_recon_loss

            # Gradient update on Discriminator network
            # torch.nn.utils.clip_grad_norm_(list(self.D.parameters()), 1.0)
            if L_gan.detach().item() < 3.5:

                self.optim_D.zero_grad()
                L_d.backward(retain_graph=True)
                self.optim_D.step()


            # Train G
            self.optim_G.zero_grad()
            L_g.backward()
            self.optim_G.step()

        losses = {
                "D_loss": L_d.detach().item(),
                "Gx_gan_loss": L_gx_gan.detach().item(),
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
            loss = loss/latent_size
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

    def add_instance_noise(self, x):
        noise_factor = self.d_img_noise_std * \
                       (1 if not self.decrease_noise else 1 - (self.current_epoch / self.epochs))
        return x + torch.randn_like(x) * noise_factor