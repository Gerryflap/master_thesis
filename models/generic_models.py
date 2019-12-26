"""
    These models can be used by giving them modules that lay out the model
"""
import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.initialization import weights_init


class Generator(torch.nn.Module):
    def __init__(self, G, latent_size, untied_bias_size=None, output_activation=None):
        """

        :param G: The generator network/module
        :param latent_size: Size of the latent space
        :param untied_bias_size: Size of the added untied biases
        """
        super().__init__()
        self.G = G
        self.latent_size = latent_size
        if untied_bias_size is not None:
            self.output_bias = torch.nn.Parameter(torch.zeros((3, 64, 64)), requires_grad=True)
        else:
            self.output_bias = None

        self.output_activ = output_activation

    def forward(self, x):
        if self.output_bias is not None:
            outp = self.G(x) + self.output_bias
        else:
            outp = self.G(x)
        if self.output_activ is not None:
            return self.output_activ(outp)
        else:
            return outp

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
        self.model = E
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


class Discriminator(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, x):
        return self.D(x)

    def init_weights(self):
        self.apply(weights_init)


class ALIDiscriminator(torch.nn.Module):
    def __init__(self, latent_size, Dx, Dz, Dxz):
        """
        Constructs the generic ALI discriminator
        :param latent_size: The used latent size
        :param Dx: A module that converts input images to vectors
        :param Dz: A module that converts incoming vectors of latent_size to outgoing vectors
        :param Dxz: A module that takes an input the size of the output of Dx and Dy concatenated
            and outputs prediction logits
        """
        super().__init__()
        self.latent_size = latent_size
        self.Dx = Dx
        self.Dz = Dz
        self.Dxz = Dxz

    def forward(self, inp):
        x, z = inp

        h_x = self.Dx(x)
        h_z = self.Dz(z)

        h = torch.cat([h_x, h_z], dim=1)

        return self.Dxz(h)

    def init_weights(self):
        self.apply(weights_init)