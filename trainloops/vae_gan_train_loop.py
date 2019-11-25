"""
    Models the VAE/GAN trainloop. It has been reordered in some places to allow for more efficient computation and to
    fix batch normalization in D
"""
import math

import torch

from trainloops.train_loop import TrainLoop


class VAEGANTrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_Gz, optim_Gx, optim_D, dataloader, cuda=False, epochs=1,
                 gamma=1e-3, max_steps_per_epoch=None, real_label_value=1.0):
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

        self.gan_labels = torch.zeros((self.batch_size*3, 1))
        self.gan_labels[:self.batch_size] = real_label_value
        self.gan_labels.requires_grad = False
        if self.cuda:
            self.gan_labels = self.gan_labels.cuda()

        self.gamma = gamma

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
            L_prior = 0.5 * torch.sum(torch.exp(z_logvar) + torch.pow(z_mean, 2) - z_logvar - 1, dim=[0, 1])

            # X_tilde <- Dec(Z)
            x_tilde = self.Gx(z)

            # Sample random Z_p values
            z_p = self.generate_z_batch(self.batch_size)

            # X_p <- Dec(Z_p)
            x_p = self.Gx(z_p)

            # Compute the activations for all Dis() computations in one go since it's more efficient and
            #   works better with batch normalization
            discriminator_inputs = torch.cat([x, x_tilde, x_p], dim=0)

            dis_preds, disl_activations = self.D(discriminator_inputs)

            disl_x = disl_activations[:self.batch_size]
            disl_x_tilde = disl_activations[self.batch_size:self.batch_size * 2]

            # Compute L_disl_llike
            L_disl_llike = self.compute_disl_llike(disl_x_tilde, disl_x)

            # Compute L_GAN
            L_GAN = torch.nn.functional.binary_cross_entropy_with_logits(dis_preds, self.gan_labels)

            # Define losses
            L_Gz = L_prior + L_disl_llike
            L_Gx = self.gamma * L_disl_llike - L_GAN
            L_D = L_GAN

            # Compute Gradients and perform updates
            self.optim_Gz.zero_grad()
            L_Gz.backward(retain_graph=True)
            self.optim_Gz.step()

            self.optim_Gx.zero_grad()
            L_Gx.backward(retain_graph=True)
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
                "L_prior": L_prior.detach().item(),
                "L_disl_llike": L_disl_llike.detach().item(),
                "L_GAN": L_GAN.detach().item(),
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
    def compute_disl_llike(pred, target):
        const = -0.5*math.log(2*math.pi, math.e)
        loss = 0.5 * (pred - target).pow(2) + const
        return loss.sum()