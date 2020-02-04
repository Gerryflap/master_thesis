import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.activations import mish, LocalResponseNorm
from util.torch.initialization import weights_init


class Encoder28(MorphingEncoder):
    def __init__(self, latent_size, h_size, use_mish=False, n_channels=1, deterministic=False, cap_variance=True, add_dense_layer=False, use_lr_norm=False):
        super().__init__()

        self.n_channels = n_channels
        self.deterministic = deterministic
        self.cap_variance = cap_variance

        if use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.latent_size = latent_size
        self.h_size = h_size
        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=4, stride=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=5, stride=2, bias=False)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, bias=False)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=4, stride=1, bias=False)

        if not use_lr_norm:
            self.bn_1 = torch.nn.BatchNorm2d(self.h_size)
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)
        else:
            self.bn_1 = LocalResponseNorm()
            self.bn_2 = LocalResponseNorm()
            self.bn_3 = LocalResponseNorm()
            self.bn_4 = LocalResponseNorm()

        self.mean_fc = torch.nn.Linear(h_size * 4, latent_size, bias=True)
        if not deterministic:
            self.std_fc = torch.nn.Linear(h_size * 4, latent_size, bias=True)

        self.dense = None
        self.add_dense_layer = add_dense_layer
        if add_dense_layer:
            self.dense = torch.nn.Linear(h_size * 4, h_size * 4, bias=False)
            self.bn_dense = torch.nn.BatchNorm1d(self.h_size * 4)



    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x = self.conv_1(inp)
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

        # Flatten to vector
        x = x.view(-1, self.h_size * 4)

        if self.add_dense_layer:
            x = self.dense(x)
            x = self.bn_dense(x)
            x = self.activ(x)

        means = self.mean_fc(x)
        if self.deterministic:
            return means, means, torch.ones_like(means)*-30
        else:
            log_vars = self.std_fc(x)
            if self.cap_variance:
                log_vars = -torch.nn.functional.softplus(log_vars)
            return self.sample(means, log_vars), means, log_vars

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds

    def init_weights(self):
        self.apply(weights_init)