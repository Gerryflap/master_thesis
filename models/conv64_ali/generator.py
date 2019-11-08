import torch
from util.torch.activations import mish


class Generator64(torch.nn.Module):
    def __init__(self, latent_size, h_size, use_mish=False, n_channels=3):
        super().__init__()

        self.n_channels = n_channels

        self.latent_size = latent_size
        self.h_size = h_size
        if use_mish:

            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.conv_1 = torch.nn.ConvTranspose2d(self.latent_size, self.h_size * 8, 4, bias=False)
        self.conv_2 = torch.nn.ConvTranspose2d(self.h_size * 8, self.h_size * 4, kernel_size=7, stride=2, bias=False)
        self.conv_3 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size*4, kernel_size=5, stride=2, bias=False)
        self.conv_4 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size*2, kernel_size=7, stride=2, bias=False)
        self.conv_5 = torch.nn.ConvTranspose2d(self.h_size*2, self.h_size*1, kernel_size=2, stride=1, bias=False)
        self.conv_6 = torch.nn.Conv2d(self.h_size, self.n_channels, kernel_size=1, stride=1, bias=True)


        self.bn_1 = torch.nn.BatchNorm2d(self.h_size * 8)
        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_5 = torch.nn.BatchNorm2d(self.h_size)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

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
        x = self.bn_4(x)
        x = self.activ(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.activ(x)

        x = self.conv_6(x)
        x = torch.tanh(x)

        return x
