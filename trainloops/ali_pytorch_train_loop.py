"""
    Models a train loop for ALI: Adversarially Learned Inference (https://arxiv.org/abs/1606.00704)
    The code for this algorithm is taken (and largely rewritten) from https://github.com/edgarriba/ali-pytorch
    This version of ALI exists as a sanity check


"""
import torch
import torch.nn.functional as F
from torch import nn

from trainloops.train_loop import TrainLoop


class ALITrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1, morgan_alpha=0.0, d_img_noise_std=0.0, d_real_label=1.0):
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

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        def tocuda(x):
            if self.cuda:
                return x.cuda()
            return x

        criterion = nn.BCEWithLogitsLoss()

        i = 0
        for (data, target) in self.dataloader:

            real_label = tocuda(torch.ones((self.batch_size, 1)))
            fake_label = tocuda(torch.zeros((self.batch_size, 1)))

            noise1 = tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (self.epochs - self.current_epoch) / self.epochs))
            noise2 = tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (self.epochs - self.current_epoch) / self.epochs))


            if data.size()[0] != self.batch_size:
                continue

            d_real = tocuda(data)

            z_fake = tocuda(torch.randn(self.batch_size, self.Gx.latent_size))
            d_fake = self.Gx(z_fake)

            output_z, _, _ = self.Gz(d_real)

            output_real = self.D((d_real + noise1, output_z))
            output_fake = self.D((d_fake + noise2, z_fake))

            loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)
            loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)

            if loss_g.data.item() < 3.5:
                self.optim_D.zero_grad()
                loss_d.backward(retain_graph=True)
                self.optim_D.step()

            self.optim_G.zero_grad()
            loss_g.backward()
            self.optim_G.step()


            i += 1

        losses = {
                "D_loss": loss_d.detach().item(),
                "G_loss": loss_g.detach().item(),
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

    @staticmethod
    def morgan_pixel_loss(x_recon, target):
        absolute_errors = torch.abs(x_recon - target)
        # WxH = float(int(absolute_errors.size()[2]) * int(absolute_errors.size()[3]))
        # loss = absolute_errors.sum()/WxH
        loss = absolute_errors.mean()
        return loss