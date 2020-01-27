"""
    Aims to reproduce the Encoder from the ALI paper.

    The deterministic mode has been added, but is not part of ALI.
    In this mode the encoder predicts the z directly instead of returning a distribution.
"""

import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.activations import mish, map_to_hypersphere
from util.torch.initialization import weights_init


class FRS64(torch.nn.Module):
    def __init__(self, latent_size=512, h_size=64, use_mish=False, n_channels=3, batchnorm=False):
        super().__init__()

        self.n_channels = n_channels

        if use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.latent_size = latent_size
        self.h_size = h_size

        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=2, stride=1, bias=not batchnorm)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=7, stride=2, bias=not batchnorm)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, bias=not batchnorm)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=7, stride=2, bias=not batchnorm)
        self.conv_5 = torch.nn.Conv2d(h_size * 4, h_size * 8, kernel_size=4, stride=1, bias=not batchnorm)

        self.bn = batchnorm
        if self.bn:
            self.bn_1 = torch.nn.BatchNorm2d(self.h_size)
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_5 = torch.nn.BatchNorm2d(self.h_size * 8)

        self.out_fc = torch.nn.Linear(h_size * 8, latent_size, bias=True)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x = self.conv_1(inp)
        if self.bn:
            x = self.bn_1(x)
        x = self.activ(x)

        x = self.conv_2(x)
        if self.bn:
            x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        if self.bn:
            x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        if self.bn:
            x = self.bn_4(x)
        x = self.activ(x)

        x = self.conv_5(x)
        if self.bn:
            x = self.bn_5(x)
        x = self.activ(x)

        # Flatten to vector
        x = x.view(-1, self.h_size * 8)

        out = self.out_fc(x)
        out = map_to_hypersphere(out)
        return out

    def init_weights(self):
        self.apply(weights_init)