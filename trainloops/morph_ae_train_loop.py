"""
    An experimental trainloop that borrows ideas from VAE/GAN,
    but lets go of the constraint that the Generator should be able to generate a face from
    any point in the latent space.
"""
import math

import torch

from trainloops.train_loop import TrainLoop


class MorphAETrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_Gz, optim_Gx, optim_D, dataloader, cuda=False, epochs=1,
                max_steps_per_epoch=None, real_label_value=1.0):
        """
        Initializes the Morph AE Trainloop
        :param listeners: A list of listeners
        :param Gz: The encoder model (forward should output z, z_mean, z_logvar)
        :param Gx: The decoder model (forward should output x)
        :param D: The discriminator model (forward should output Dis(x) and Dis_l(x)
        :param optim_Gz: The encoder optimizer
        :param optim_Gx: The decoder optimizer
        :param optim_D: The discriminator optimizer
        :param dataloader: The dataloader used
        :param cuda: Whether cuda should be used
        :param epochs: The number of epochs to run for
        :param max_steps_per_epoch: Ends an epoch at this amount of steps, instead of when the dataloader is done.
            The dataloader is NOT reset, so it might continue in the next epoch which could result in the epoch ending
            when the dataloader is done instead of when the max_steps is reached.
        :param real_label_value: Gives a value to the "real" label of the discriminator.
            Using  0.9 might aide convergence according to https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
        """
        super().__init__(listeners, epochs)
        self.real_label_value = real_label_value
        self.batch_size = dataloader.batch_size
        self.Gz = Gz
        self.Gx = Gx
        self.D = D
        self.optim_Gx = optim_Gx
        self.optim_Gz = optim_Gz
        self.optim_D = optim_D
        self.dataloader = dataloader
        self.cuda = cuda
        self.max_steps_per_epoch = max_steps_per_epoch

        self.label_real = torch.ones((self.batch_size, 1))
        self.label_real_d = torch.ones((self.batch_size, 1)) * real_label_value
        self.label_fake = torch.zeros((self.batch_size, 1))

        if self.cuda:
            self.label_real = self.label_real.cuda()
            self.label_real_d = self.label_real_d.cuda()
            self.label_fake = self.label_fake.cuda()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()


    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        for i, (x1, x2) in enumerate(self.dataloader):
            if self.max_steps_per_epoch is not None and i >= self.max_steps_per_epoch:
                break

            if x1.size()[0] != self.batch_size:
                continue
            # Draw M (= batch_size) samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            # Z <- Enc(X)
            z_morph, z1, z2 = self.Gz.morph(x1, x2, return_all=True)

            z_batch = torch.cat([z1, z2], dim=0)

            z_batch_mean, z_batch_var = z_batch.mean(dim=0), z_batch.var(dim=0)

            x1_rec = self.Gx(z1)
            x2_rec = self.Gx(z2)
            x_morph = self.Gx(z_morph)

            dis_x1, disl_x1 = self.D(x1)
            dis_x2, disl_x2 = self.D(x2)
            dis_x1r, disl_x1r = self.D(x1_rec)
            dis_x2r, disl_x2r = self.D(x2_rec)
            dis_xm, disl_xm = self.D(x_morph)

            # Compute L_GAN
            L_GAN_generated = (
                                  self.loss_fn(dis_x1r, self.label_fake) +
                                  self.loss_fn(dis_x2r, self.label_fake) +
                                  self.loss_fn(dis_xm, self.label_fake)
                              )/3

            # The GAN loss for G is different when label smoothing is used.
            # The labels are not inverted since -1*L_GAN is still used for Gx
            L_GAN_real = 0.5 * (self.loss_fn(dis_x1, self.label_real_d) + self.loss_fn(dis_x2, self.label_real_d))
            L_D = L_GAN_generated + L_GAN_real

            L_dis_x1 = self.disl_loss_fn(disl_x1r, disl_x1)
            L_dis_x2 = self.disl_loss_fn(disl_x2r, disl_x2)
            L_dis_x = L_dis_x1 + L_dis_x2
            L_dis_morph = self.disl_loss_fn(disl_xm, disl_x1) + self.disl_loss_fn(disl_xm, disl_x2)

            # L_latent = torch.nn.functional.mse_loss(z_batch_mean, torch.zeros_like(z_batch_mean)) +\
            #     torch.nn.functional.mse_loss(z_batch_var, torch.ones_like(z_batch_mean))

            L_g = L_dis_x1 + L_dis_x2 + L_dis_morph  # + L_latent

            # Compute Gradients and perform updates
            self.optim_Gz.zero_grad()
            L_g.backward(retain_graph=True)
            self.optim_Gz.step()

            self.optim_Gx.zero_grad()
            L_g.backward(retain_graph=True)
            self.optim_Gx.step()

            self.optim_D.zero_grad()
            L_D.backward()
            self.optim_D.step()

        self.Gz.eval()
        self.Gx.eval()
        self.D.eval()

        return {
            "epoch": self.current_epoch,
            "losses": {
                # "L_latent": L_latent.detach().item(),
                "L_dis_x": L_dis_x.detach().item(),
                "L_dis_morph": L_dis_morph.detach().item(),
                "L_G": L_g.detach().item(),
                "L_D": L_D.detach().item(),
            },
            "networks": {
                "Gx": self.Gx,
                "Gz": self.Gz,
                "D": self.D,
            },
            "optimizers": {
                "Gz_optimizer": self.optim_Gz,
                "Gx_optimizer": self.optim_Gx,
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
    def disl_loss_fn(pred, target):
        return torch.nn.functional.mse_loss(pred, target)

    @staticmethod
    def compute_disl_llike(pred, target):
        const = -0.5*math.log(2*math.pi, math.e)
        loss = 0.5 * (pred - target).pow(2) + const
        return loss.sum()