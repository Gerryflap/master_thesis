import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.activations import mish, map_to_hypersphere
from util.torch.initialization import weights_init


class FRS28(torch.nn.Module):
    def __init__(self, latent_size, h_size, use_mish=False, n_channels=1, add_dense_layer=False):
        super().__init__()

        self.n_channels = n_channels

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

        self.bn_1 = torch.nn.BatchNorm2d(self.h_size)
        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)

        self.out_fc = torch.nn.Linear(h_size * 4, latent_size, bias=True)

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

        out = self.out_fc(x)
        out = map_to_hypersphere(out)
        return out

    def init_weights(self):
        self.apply(weights_init)
