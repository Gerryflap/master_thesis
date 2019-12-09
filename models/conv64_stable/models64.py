"""
    This code is a slightly modified version of the model.py file on https://github.com/edgarriba/ali-pytorch

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.morphing_encoder import MorphingEncoder
from util.torch.initialization import weights_init


class Generator(nn.Module):

    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size

        self.output_bias = nn.Parameter(torch.zeros(3, 64, 64), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 1024, 4, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, 4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 1, stride=1, bias=False)
        )

    def forward(self, input):
        input = input.view((input.size(0), input.size(1), 1, 1))
        output = self.main(input)
        output = F.sigmoid(output + self.output_bias)
        return output

    def init_weights(self):
        self.apply(weights_init)


class Encoder(MorphingEncoder):

    def __init__(self, latent_size, noise=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        if noise:
            self.latent_size *= 2
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512, 1024, 4, stride=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(1024, 2048, 3, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(2048, 4096, 1, stride=1, bias=False),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(4096, self.latent_size, 1, stride=1, bias=True)
        )

    def forward(self, input):
        batch_size = input.size()[0]
        output = self.main(input)
        output = output.view(batch_size, self.latent_size)

        means, log_var = output[:, :self.latent_size//2], output[:, self.latent_size//2:]

        return self.sample(means, log_var), means, log_var

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds

    def init_weights(self):
        self.apply(weights_init)


class Discriminator(nn.Module):

    def __init__(self, latent_size, dropout, output_size=10):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size

        self.infer_x = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 1024, 4, stride=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 3, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.latent_size, 1024, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(2048, 2048, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(2048, 2048, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(2048, self.output_size, 1, stride=1, bias=True)

    def forward(self, xz):
        x, z = xz
        z = z.view((z.size(0), z.size(1), 1, 1))
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        # if self.output_size == 1:
        #     output = F.sigmoid(output)
        return output.view((x.size(0), 1))

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    G = Generator(10)
    E = Encoder(10, noise=True)
    D = Discriminator(10, 0.2, 1)

    z = torch.normal(0, 1, (64, 10))
    x = G(z)
    z_hat = E(x)
    d = D((x, z))
    print(d)