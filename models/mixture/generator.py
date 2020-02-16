import torch

from util.torch.activations import LocalResponseNorm
from util.torch.initialization import weights_init


class Generator(torch.nn.Module):
    def __init__(self, latent_size, h_size=64, tanh=True, bn=True, lr_norm=False):
        super().__init__()
        self.latent_size = latent_size
        self.h_size = h_size
        self.tanh = tanh

        if bn:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(latent_size, h_size),
                torch.nn.BatchNorm1d(h_size),
                torch.nn.LeakyReLU(0.02),

                torch.nn.Linear(h_size, h_size),
                torch.nn.BatchNorm1d(h_size),
                torch.nn.LeakyReLU(0.02),

                torch.nn.Linear(h_size, 2),
            )
        elif lr_norm:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(latent_size, h_size),
                torch.nn.LeakyReLU(0.02),
                LocalResponseNorm(),

                torch.nn.Linear(h_size, h_size),
                torch.nn.LeakyReLU(0.02),
                LocalResponseNorm(),

                torch.nn.Linear(h_size, 2),
            )
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(latent_size, h_size),
                torch.nn.LeakyReLU(0.02),

                torch.nn.Linear(h_size, h_size),
                torch.nn.LeakyReLU(0.02),

                torch.nn.Linear(h_size, 2),
            )

    def forward(self, inp):
        if self.tanh:
            return torch.tanh(self.model(inp))
        else:
            return self.model(inp)

    def init_weights(self):
        self.apply(weights_init)
