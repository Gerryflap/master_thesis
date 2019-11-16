"""
    Models a train loop for ALI: Adversarially Learned Inference (https://arxiv.org/abs/1606.00704)
    Additionally, this train loop can also perform the MorGAN algorithm by setting the MorGAN alpha
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop


class ALITrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim, dataloader, cuda=False, epochs=1, morgan_alpha=0.0):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.Gz = Gz
        self.Gx = Gx
        # self.G = torch.nn.ModuleList([self.Gx, self.Gz])
        self.D = D
        self.optim = optim
        self.dataloader = dataloader
        self.cuda = cuda
        self.morgan_alpha = morgan_alpha
        self.morgan = morgan_alpha != 0

    @staticmethod
    def set_requires_grad(model, value):
        for param in model.parameters():
            param.requires_grad = value

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        for i, (x, _) in enumerate(self.dataloader):
            if x.size()[0] != self.batch_size:
                continue
            # Draw M (= batch_size) samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x = x.cuda()
            z = self.generate_z_batch(self.batch_size)

            # Sample from conditionals (sampling is implemented by models)
            z_hat = self.Gz.encode(x)
            x_tilde = self.Gx(z)

            # Compute discriminator predictions
            self.set_requires_grad(self.D, False)
            self.D.eval()
            dis_q_g = self.D((x, z_hat))
            dis_p_g = self.D((x_tilde, z))

            self.D.train()
            self.set_requires_grad(self.D, True)

            z_hat = z_hat.detach()
            x_tilde = x_tilde.detach()
            dis_q_d = self.D((x, z_hat))
            dis_p_d = self.D((x_tilde, z))


            # Compute Discriminator loss
            L_d = F.binary_cross_entropy_with_logits(dis_q_d, torch.ones_like(dis_q_d), reduction="mean") + \
                  F.binary_cross_entropy_with_logits(dis_p_d, torch.zeros_like(dis_p_d), reduction="mean")

            L_gz = F.binary_cross_entropy_with_logits(dis_q_g, torch.zeros_like(dis_q_g), reduction="mean")
            L_gx = F.binary_cross_entropy_with_logits(dis_p_g, torch.ones_like(dis_p_g), reduction="mean")
            L_g = L_gz + L_gx

            # Extra code for the MorGAN algorithm. This is not part of ALI
            if self.morgan:
                x_recon = self.Gx(z_hat)
                L_pixel = self.morgan_pixel_loss(x_recon, x)
                L_syn = L_g + self.morgan_alpha * L_pixel

            # Gradient update on all networks
            self.optim.zero_grad()

            L_total = L_d + (L_g if not self.morgan else L_syn)

            L_total.backward()
            self.optim.step()

        self.Gx.eval()
        self.Gz.eval()
        self.D.eval()

        print(list(self.Gx.conv_1.parameters())[0].mean())

        return {
            "epoch": self.current_epoch,
            "losses": {
                "D_loss: ": L_d.detach().item(),
                "G_loss: ": L_g.detach().item(),
            },
            "networks": {
                "Gx": self.Gx,
                "Gz": self.Gz,
                "D": self.D,
            },
            "optimizers": {
                "optimizer": self.optim,
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