"""
    Aims to reproduce the Encoder from the ALI paper.
"""

import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.activations import mish


class Encoder64(MorphingEncoder):
    def __init__(self, latent_size=512, h_size=64, use_mish=False, n_channels=3):
        super().__init__()

        self.n_channels = n_channels

        if use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.latent_size = latent_size
        self.h_size = h_size

        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=2, stride=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=7, stride=2, bias=False)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, bias=False)
        self.conv_3 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=7, stride=2, bias=False)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 8, kernel_size=4, stride=1, bias=False)

        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_5 = torch.nn.BatchNorm2d(self.h_size * 8)

        self.mean_fc = torch.nn.Linear(h_size * 8, latent_size, bias=True)
        self.std_fc = torch.nn.Linear(h_size * 8, latent_size, bias=True)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.activ(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.activ(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.activ(x)

        # Flatten to vector
        x = x.view(-1, self.h_size * 8)

        means = self.mean_fc(x)
        log_vars = self.std_fc(x)
        return self.sample(means, log_vars), means, log_vars

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds
