import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.activations import mish
from util.torch.initialization import weights_init


class VAEGANEncoder64(MorphingEncoder):
    def __init__(self, latent_size, h_size, n_channels=1):
        super().__init__()

        self.n_channels = n_channels


        self.activ = mish

        self.latent_size = latent_size
        self.h_size = h_size
        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=5,  stride=2, padding=2, bias=False)
        # 32x32
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=5, stride=2, padding=2, bias=False)
        # 16x16
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, padding=2, bias=False)
        # 8x8x256

        self.bn_1 = torch.nn.BatchNorm2d(self.h_size)
        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_fc = torch.nn.BatchNorm1d(self.h_size * 32)

        self.fc = torch.nn.Linear(8*8*h_size*4, h_size*32, bias=False)

        self.mean_fc = torch.nn.Linear(h_size * 32, latent_size)
        self.std_fc = torch.nn.Linear(h_size * 32, latent_size)



    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.activ(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.activ(x)

        # Flatten to vector
        x = x.view(-1, 8 * 8 * self.h_size * 4)

        x = self.fc(x)
        x = self.bn_fc(x)
        x = self.activ(x)

        means = self.mean_fc(x)
        log_vars = self.std_fc(x)
        return self.sample(means, log_vars), means, log_vars

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds

    def init_weights(self):
        self.apply(weights_init)