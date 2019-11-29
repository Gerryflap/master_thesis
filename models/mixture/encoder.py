import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.initialization import weights_init


class Encoder(MorphingEncoder):
    def __init__(self, latent_size, h_size=64):
        super().__init__()
        self.latent_size = latent_size
        self.h_size = h_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, h_size),
            torch.nn.BatchNorm1d(h_size),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(h_size, h_size),
            torch.nn.BatchNorm1d(h_size),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(h_size, latent_size*2),
        )

    def forward(self, inp):
        model_out = self.model(inp)

        # Split the output of the model into two parts, means and log variances
        z_mean, z_logvar = model_out[:, :self.latent_size], model_out[:, self.latent_size:]

        return self.sample(z_mean, z_logvar), z_mean, z_logvar

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds

    def init_weights(self):
        self.apply(weights_init)