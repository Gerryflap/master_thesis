import torch
from util.torch.activations import mish


class Generator28(torch.nn.Module):
    def __init__(self, latent_size, h_size, use_mish=False, bias=True, n_channels=1):
        super().__init__()

        self.n_channels = n_channels

        self.latent_size = latent_size
        self.h_size = h_size
        self.bias = bias
        if use_mish:

            self.activ = mish
        else:
            self.activ = torch.relu

        self.conv_1 = torch.nn.ConvTranspose2d(self.latent_size, self.h_size * 4, 4, bias=self.bias)
        self.conv_2 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size * 2, kernel_size=5, stride=2, bias=self.bias)
        self.conv_3 = torch.nn.ConvTranspose2d(self.h_size * 2, self.h_size, kernel_size=5, stride=2, bias=self.bias)
        self.conv_4 = torch.nn.ConvTranspose2d(self.h_size, n_channels, kernel_size=4, stride=1, bias=self.bias)

        self.bn_1 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size)

    def forward(self, inp):
        x = inp.view(-1, self.latent_size, 1, 1)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activ(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        x = torch.tanh(x)

        return x
