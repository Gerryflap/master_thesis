"""
    These models can be used by giving them modules that lay out the model
"""
import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.initialization import weights_init


class Generator(torch.nn.module):
    def __init__(self, G, latent_size):
        self.G = G
        self.latent_size = latent_size

    def forward(self, x):
        return self.G(x)

    def init_weights(self):
        self.apply(weights_init)


class Encoder(MorphingEncoder):
    def __init__(self, E, latent_size, deterministic=False):
        """
        Creates the encoder
        :param E: A torch module that outputs (batch_size, latent_size) or (batch_size, latent_size*2) sized tensors.
            The second is used in non-deterministic mode
        :param latent_size: Size of the latent space, this value might be expected here by some algorithms
        :param deterministic: Whether the encoder is deterministic or stochastic
        """
        super().__init__()
        self.E = E
        self.latent_size = latent_size
        self.deterministic = deterministic

    def forward(self, inp):
        model_out = self.model(inp)

        if self.deterministic:
            # Give deterministic sample and simulate a very low "variance" for interfaces that might need it
            return model_out, model_out, torch.ones_like(model_out)*-30.0
        else:

            # Split the output of the model into two parts, means and log variances
            z_mean, z_logvar = model_out[:, :self.latent_size], model_out[:, self.latent_size:]
            z_logvar = -torch.nn.functional.softplus(z_logvar)
            return self.sample(z_mean, z_logvar), z_mean, z_logvar

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds

    def init_weights(self):
        self.apply(weights_init)


class ALIDiscriminator(torch.nn.Module):
    def __init__(self, Dx, Dz, Dxz):
        super().__init__()
