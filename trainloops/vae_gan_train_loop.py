"""
    Models the VAE/GAN trainloop. It has been reordered in some places to allow for more efficient computation and to
    fix batch normalization in D
"""
import math

import torch
from torch.distributions import Normal, MultivariateNormal
from torch.optim.lr_scheduler import ExponentialLR

from trainloops.train_loop import TrainLoop


class VAEGANTrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_Gz, optim_Gx, optim_D, dataloader, cuda=False, epochs=1,
                 gamma=1e-3, max_steps_per_epoch=None, real_label_value=1.0, beta=1.0, lr_decay=1.0, feed_reconstructions_into_D=True):
        """
        Initializes the VAE/GAN Trainloop
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
        :param gamma: The gamma parameter used in VAE/GAN, scales L_disl_llike in the decoder
        :param max_steps_per_epoch: Ends an epoch at this amount of steps, instead of when the dataloader is done.
            The dataloader is NOT reset, so it might continue in the next epoch which could result in the epoch ending
            when the dataloader is done instead of when the max_steps is reached.
        :param real_label_value: Gives a value to the "real" label of the discriminator.
            Using  0.9 might aide convergence according to https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
        :param beta: Scales L_prior
        :param lr_decay: The lr decay used in the exponential decay.
        :param feed_reconstructions_into_D: When true, feed x_tilde into D.
        """
        super().__init__(listeners, epochs)
        self.feed_reconstructions_into_D = feed_reconstructions_into_D
        self.real_label_value = real_label_value
        self.batch_size = dataloader.batch_size
        self.Gz = Gz
        self.Gx = Gx
        self.D = D

        self.lr_decay = lr_decay
        self.optim_Gx = optim_Gx
        self.optim_Gz = optim_Gz
        self.optim_D = optim_D

        if self.lr_decay != 1.0:
            self.lr_Gz = ExponentialLR(self.optim_Gz, gamma=self.lr_decay)
            self.lr_Gx = ExponentialLR(self.optim_Gx, gamma=self.lr_decay)
            self.lr_D = ExponentialLR(self.optim_D, gamma=self.lr_decay)


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
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

        self.gamma = gamma
        self.beta = beta

        self.margin = 0.35
        self.equilibrium = 0.68

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        for i, (x, _) in enumerate(self.dataloader):
            if self.max_steps_per_epoch is not None and i >= self.max_steps_per_epoch:
                break

            if x.size()[0] != self.batch_size:
                continue
            # Draw M (= batch_size) samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x = x.cuda()

            # Z <- Enc(X)
            z, z_mean, z_logvar = self.Gz(x)

            # L_prior <- Dkl(q(Z|X)||p(Z))
            # We're already using log variance instead of the standard deviation,
            #   so the equation might not look like the one in the thesis. They should be equivalent.
            # L_prior = 0.5 * torch.sum(torch.exp(z_logvar) + torch.pow(z_mean, 2) - z_logvar - 1, dim=[0, 1]) \
            #         / self.batch_size

            L_prior = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())/self.batch_size

            # X_tilde <- Dec(Z)
            x_tilde = self.Gx(z)

            # Sample random Z_p values
            z_p = self.generate_z_batch(self.batch_size)

            # X_p <- Dec(Z_p)
            x_p = self.Gx(z_p)

            # Combine the D inputs (this is not advisable in normal GANs, but the official VAE/GAN implementation seems to do this)
            d_inputs = torch.cat([x, x_tilde, x_p])
            d_outputs, _ = self.D(d_inputs)

            dis_x = d_outputs[:self.batch_size]
            dis_x_tilde = d_outputs[self.batch_size:self.batch_size*2]
            dis_xp = d_outputs[self.batch_size*2:]

            # Combine x_tilde and x into one batch and compute dis_l for both
            d_inputs = torch.cat([x, x_tilde])
            _, d_outputs = self.D(d_inputs)
            disl_x, disl_x_tilde = d_outputs[:self.batch_size], d_outputs[self.batch_size:]

            # Compute L_disl_llike
            L_disl_llike = self.compute_disl_llike(disl_x_tilde, disl_x) / self.batch_size

            # Compute L_GAN
            L_GAN_x_p = self.loss_fn(dis_xp, self.label_fake)
            if self.feed_reconstructions_into_D:
                L_GAN_x_tilde = self.loss_fn(dis_x_tilde, self.label_fake)
                L_GAN_generated = (L_GAN_x_tilde + L_GAN_x_p) * 0.5
            else:
                L_GAN_generated = L_GAN_x_p

            # The GAN loss for G is different when label smoothing is used.
            # The labels are not inverted since -1*L_GAN is still used for Gx
            L_GAN_d = (self.loss_fn(dis_x, self.label_real_d) + L_GAN_generated)/self.batch_size
            L_GAN_g = (self.loss_fn(dis_x, self.label_real) + L_GAN_generated)/self.batch_size

            # Define losses
            L_Gz = L_prior + L_disl_llike/self.beta
            L_Gx = self.gamma * L_disl_llike - L_GAN_g
            L_D = L_GAN_d

            train_d = True
            train_gx = True
            if self.equilibrium - self.margin > L_GAN_g.cpu().detach().item():
                train_d = False
            if self.equilibrium + self.margin < L_GAN_g.cpu().detach().item():
                train_gx = False
            if not (train_d or train_gx):
                train_gx = True
                train_d = True

            # Compute Gradients and perform updates
            self.optim_Gz.zero_grad()
            L_Gz.backward(retain_graph=True)
            self.optim_Gz.step()

            if train_gx:
                self.optim_Gx.zero_grad()
                L_Gx.backward(retain_graph=train_d)
                self.optim_Gx.step()

            if train_d:
                self.optim_D.zero_grad()
                L_D.backward()
                self.optim_D.step()

        if self.lr_decay != 1:
            self.lr_Gz.step(self.current_epoch)
            self.lr_Gx.step(self.current_epoch)
            self.lr_D.step(self.current_epoch)

        self.Gz.eval()
        self.Gx.eval()
        self.D.eval()

        return {
            "epoch": self.current_epoch,
            "losses": {
                "L_prior": L_prior.detach().item(),
                "L_disl_llike": L_disl_llike.detach().item(),
                "L_GAN": L_GAN_g.detach().item(),
                "L_Gz": L_Gz.detach().item(),
                "L_Gx": L_Gx.detach().item(),
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
    def compute_disl_llike_old(pred, target):
        const = -0.5 * math.log(2 * math.pi, math.e)
        loss = 0.5 * (pred - target).pow(2) - const
        return loss.sum()

    @staticmethod
    def compute_disl_llike(pred, target):
        distribution = Normal(pred, 1.0)
        loss = distribution.log_prob(target)
        return loss.sum()