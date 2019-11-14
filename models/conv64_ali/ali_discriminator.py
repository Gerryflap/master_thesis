"""
    This discriminator takes a tuple of (x, z) instead of just an x.
    Default hyper parameters aim to reproduce the official ALI hyper parameters.
"""
import torch
from util.torch.activations import mish
from util.torch.initialization import weights_init


class ALIDiscriminator64(torch.nn.Module):
    def __init__(self, latent_size=512, h_size=64, fc_h_size=None, use_bn=True, use_mish=False, n_channels=3, dropout=0.2):
        super().__init__()

        self.n_channels = n_channels
        self.latent_size = latent_size

        if use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.dropout = dropout
        self.h_size = h_size
        if fc_h_size is None:
            self.fc_h_size = h_size*16
        else:
            self.fc_h_size = fc_h_size

        # In the official github repository, the biases are disabled for many layers.
        # I have aimed to reproduce this, although tied/untied weights are not reproduced and kept on default.

        # Convolutional layers:
        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=2,  stride=1, bias=True)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=7, stride=2, bias=False)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, bias=False)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=7, stride=2, bias=False)
        self.conv_5 = torch.nn.Conv2d(h_size * 4, h_size * 8, kernel_size=4, stride=1, bias=False)

        self.use_bn = use_bn
        if use_bn:
            # In the paper, the first layer is also supposed to have batch normalization.
            # This lead to bad preliminary results for me and the official ALI code also does not do this.
            # For this reason the first layer will not have batch normalization.

            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2, 0.8)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4, 0.8)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4, 0.8)
            self.bn_5 = torch.nn.BatchNorm2d(self.h_size * 8, 0.8)

        # Linear layers (1x1 convolutions in the paper)
        if dropout != 0:
            self.dropout_layer = torch.nn.Dropout(dropout, False)

        self.lin_z1 = torch.nn.Linear(latent_size, self.fc_h_size, bias=False)
        self.lin_z2 = torch.nn.Linear(self.fc_h_size, self.fc_h_size, bias=False)

        self.lin_xz1 = torch.nn.Linear(h_size*8 + self.fc_h_size, self.fc_h_size*2, bias=True)
        self.lin_xz2 = torch.nn.Linear(self.fc_h_size*2, self.fc_h_size*2, bias=True)
        self.lin_xz3 = torch.nn.Linear(self.fc_h_size*2, 1, bias=True)

        # Initialize weights
        self.apply(weights_init)


    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x, z = inp

        h_x = self.compute_dx(x)
        h_z = self.compute_dz(z)

        prediction = self.compute_dxz(h_x, h_z)
        # There is no sigmoid applied to the output here.
        # This is done in the loss function for improved numerical stability
        # While this does make the code more confusing, it drastically improves the stability in practice.
        return prediction

    def compute_dx(self, x):
        h = self.conv_1(x)
        h = self.activ(h)

        h = self.conv_2(h)
        if self.use_bn:
            h = self.bn_2(h)
        h = self.activ(h)

        h = self.conv_3(h)
        if self.use_bn:
            h = self.bn_3(h)
        h = self.activ(h)

        h = self.conv_4(h)
        if self.use_bn:
            h = self.bn_4(h)
        h = self.activ(h)

        h = self.conv_5(h)
        if self.use_bn:
            h = self.bn_5(h)
        h = self.activ(h)

        # Flatten to batch of vectors
        h_x = h.view(-1, self.h_size * 8)
        return h_x

    def compute_dz(self, z):
        if self.dropout != 0:
            z = self.dropout_layer(z)
        h_z = self.lin_z1(z)
        h_z = self.activ(h_z)

        if self.dropout != 0:
            h_z = self.dropout_layer(h_z)
        h_z = self.lin_z2(h_z)
        h_z = self.activ(h_z)

        return h_z

    def compute_dxz(self, h_x, h_z):
        h = torch.cat((h_x, h_z), dim=1)

        if self.dropout != 0:
            h = self.dropout_layer(h)
        h = self.lin_xz1(h)
        h = self.activ(h)

        if self.dropout != 0:
            h = self.dropout_layer(h)
        h = self.lin_xz2(h)
        h = self.activ(h)

        if self.dropout != 0:
            h = self.dropout_layer(h)
        h = self.lin_xz3(h)

        return h

