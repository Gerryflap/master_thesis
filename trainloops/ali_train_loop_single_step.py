"""
    Models a train loop for ALI: Adversarially Learned Inference (https://arxiv.org/abs/1606.00704)
    Additionally, this train loop can also perform the MorGAN algorithm by setting the MorGAN alpha
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop


class ALITrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1, morgan_alpha=0.0, d_img_noise_std=0.0, d_real_label=1.0, decrease_noise=True):
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
        self.morgan_alpha = morgan_alpha
        self.morgan = morgan_alpha != 0
        self.d_img_noise_std = d_img_noise_std
        self.d_real_label = d_real_label
        self.decrease_noise = decrease_noise

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

            x_no_noise = x
            z = self.generate_z_batch(self.batch_size)

            # Add noise to the inputs if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                noise_factor = self.d_img_noise_std * \
                               (1 if not self.decrease_noise else 1 - (self.current_epoch/self.epochs))
                x = x + torch.randn_like(x) * noise_factor

            # Sample from conditionals (sampling is implemented by models)
            z_hat = self.Gz.encode(x)

            dis_q = self.D((x, z_hat))
            d_real_labels = torch.ones_like(dis_q) * self.d_real_label

            x_tilde = self.Gx(z)

            # Add noise to the inputs of D if the standard deviation isn't defined to be 0
            if self.d_img_noise_std != 0.0:
                noise_factor = self.d_img_noise_std * \
                               (1 if not self.decrease_noise else (1 - (self.current_epoch/self.epochs)))
                x_tilde = x_tilde + torch.randn_like(x_tilde) * noise_factor

            dis_p = self.D((x_tilde, z))

            L_d_fake = F.binary_cross_entropy_with_logits(dis_p, torch.zeros_like(dis_q))
            L_d_real = F.binary_cross_entropy_with_logits(dis_q, d_real_labels)
            L_d = L_d_real + L_d_fake

            L_g_fake = F.binary_cross_entropy_with_logits(dis_p, torch.ones_like(dis_q))
            L_g_real = F.binary_cross_entropy_with_logits(dis_q, torch.zeros_like(dis_q))

            L_g = L_g_real + L_g_fake

            if self.morgan:
                x_recon = self.Gx(z_hat)
                L_pixel = self.morgan_pixel_loss(x_recon, x_no_noise)
                L_syn = L_g + self.morgan_alpha * L_pixel

            # Gradient update on Discriminator network
            # torch.nn.utils.clip_grad_norm_(list(self.D.parameters()), 1.0)
            if L_g.detach().item() < 3.5:

                self.optim_D.zero_grad()
                L_d.backward(retain_graph=True)
                self.optim_D.step()


            # Train G
            self.optim_G.zero_grad()
            if self.morgan:
                L_syn.backward()
            else:
                L_g.backward()
            self.optim_G.step()

        losses = {
                "D_loss": L_d.detach().item(),
                "G_loss": L_g.detach().item(),
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