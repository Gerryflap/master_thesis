"""
    Models a train loop for ALI: Adversarially Learned Inference (https://arxiv.org/abs/1606.00704)
    Additionally, this train loop can also perform the MorGAN algorithm by setting the MorGAN alpha
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop


class ALITrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1, morgan_alpha=0.0):
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
            dis_q = self.D((x, z_hat.detach()))
            L_d_real = F.binary_cross_entropy_with_logits(dis_q, torch.ones_like(dis_q), reduction="mean")
            L_d_real.backward()

            x_tilde = self.Gx(z)
            dis_p = self.D((x_tilde.detach(), z))
            L_d_fake = F.binary_cross_entropy_with_logits(dis_p, torch.zeros_like(dis_q), reduction="mean")
            L_d_fake.backward()

            L_d = L_d_real + L_d_fake

            # Gradient update on Discriminator network
            torch.nn.utils.clip_grad_norm_(list(self.D.parameters()), 1.0)
            self.optim_D.step()

            # Train G
            self.optim_G.zero_grad()

            z = self.generate_z_batch(self.batch_size)

            # Sample from conditionals (sampling is implemented by models)
            z_hat = self.Gz.encode(x)
            dis_q = self.D((x, z_hat))
            L_gz = F.binary_cross_entropy_with_logits(dis_q, torch.zeros_like(dis_q), reduction="mean")
            L_gz.backward()

            x_tilde = self.Gx(z)
            dis_p = self.D((x_tilde, z))
            L_gx = F.binary_cross_entropy_with_logits(dis_p, torch.ones_like(dis_q), reduction="mean")
            L_gx.backward()

            L_g = L_gz + L_gx

            # Extra code for the MorGAN algorithm. This is not part of ALI
            if self.morgan:
                x_recon = self.Gx(z_hat)
                L_pixel = self.morgan_alpha * self.morgan_pixel_loss(x_recon, x)
                L_syn = L_g + L_pixel

            # Gradient update on Generator networks
            if self.morgan:
                L_pixel.backward()
            torch.nn.utils.clip_grad_norm_(list(self.Gx.parameters()) + list(self.Gz.parameters()), 1.0)
            self.optim_G.step()
        self.Gx.eval()
        self.Gz.eval()
        self.D.eval()

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
        WxH = float(int(absolute_errors.size()[2]) * int(absolute_errors.size()[3]))
        loss = absolute_errors.sum()/WxH
        return loss