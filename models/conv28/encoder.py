import torch
from util.torch.activations import mish


class Encoder28(torch.nn.Module):
    def __init__(self, latent_size, h_size, use_mish=False):
        super().__init__()

        if use_mish:
            self.activ = mish
        else:
            self.activ = torch.nn.functional.relu

        self.latent_size = latent_size
        self.h_size = h_size
        self.conv_1 = torch.nn.Conv2d(1, h_size, kernel_size=4, stride=1)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=5, stride=2)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=4, stride=1)

        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)

        self.mean_fc = torch.nn.Linear(h_size * 4, latent_size)
        self.std_fc = torch.nn.Linear(h_size * 4, latent_size)

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

        # Flatten to vector
        x = x.view(-1, self.h_size * 4)

        means = self.mean_fc(x)
        log_vars = self.std_fc(x)
        return self.sample(means, log_vars), means, log_vars

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds
