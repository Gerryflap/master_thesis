"""
    This discriminator takes a tuple of (x, z) instead of just an x.
    All parameter choices are not specifically taken from ALI.
"""
import torch
from util.torch.activations import mish
from util.torch.initialization import weights_init


class ALIDiscriminator28(torch.nn.Module):
    def __init__(self, latent_size, h_size, fc_h_size=None, use_bn=False, use_mish=False, n_channels=1, dropout=0.0, use_logits=True, mbatch_stddev=False):
        super().__init__()

        self.use_logits = use_logits
        self.n_channels = n_channels
        self.latent_size = latent_size
        self.mbatch_stddev = mbatch_stddev

        if use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.dropout = dropout
        self.h_size = h_size
        if fc_h_size is None:
            self.fc_h_size = h_size*8
        else:
            self.fc_h_size = fc_h_size
        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=4,  stride=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=5, stride=2, bias=False)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, bias=False)
        self.conv_4 = torch.nn.Conv2d(h_size * 4 + (1 if mbatch_stddev else 0), h_size * 4, kernel_size=4, stride=1, bias=False)

        self.use_bn = use_bn
        if use_bn:
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)

        if dropout != 0:
            self.dropout_layer = torch.nn.Dropout(dropout, False)
            self.dropout_conv_layer = torch.nn.Dropout2d(dropout, False)

        self.lin_z1 = torch.nn.Linear(latent_size, self.fc_h_size, bias=False)
        self.lin_z2 = torch.nn.Linear(self.fc_h_size + (1 if mbatch_stddev else 0), self.fc_h_size, bias=False)

        self.lin_xz1 = torch.nn.Linear(h_size*4 + self.fc_h_size, self.fc_h_size*2, bias=True)
        self.lin_xz2 = torch.nn.Linear(self.fc_h_size*2, self.fc_h_size*2, bias=True)
        self.lin_xz3 = torch.nn.Linear(self.fc_h_size*2, 1, bias=True)

    def init_weights(self):
        self.apply(weights_init)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x, z = inp

        h_x, _ = self.compute_dx(x)
        h_z = self.compute_dz(z)

        prediction = self.compute_dxz(h_x, h_z)
        return prediction

    def compute_dx(self, x):
        h = self.conv_1(x)
        if self.dropout != 0:
            h = self.dropout_conv_layer(h)
        h = self.activ(h)

        h = self.conv_2(h)
        if self.dropout != 0:
            h = self.dropout_conv_layer(h)
        if self.use_bn:
            h = self.bn_2(h)
        h = self.activ(h)

        h = self.conv_3(h)
        h_stats = h
        if self.dropout != 0:
            h = self.dropout_conv_layer(h)
        if self.use_bn:
            h = self.bn_3(h)
        h = self.activ(h)

        dis_l = h

        if self.mbatch_stddev:
            self.add_minibatch_stddev(h, h_stats)

        h = self.conv_4(h)
        if self.dropout != 0:
            h = self.dropout_conv_layer(h)
        if self.use_bn:
            h = self.bn_4(h)
        h = self.activ(h)

        # Flatten to batch of vectors
        h_x = h.view(-1, self.h_size * 4)
        return h_x, dis_l

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

    def add_minibatch_stddev(self, h, h_stats):
        # Compute stddev over batch dim (so for every pixel/channel)
        stddevs = torch.std(h_stats, dim=0)

        # Compute mean stddev
        mean_stddev = stddevs.mean()

        # Stretch to size of h apart from channel dim and concat
        size = list(h.size())
        size[1] = 1
        mbatch_stddev = torch.ones(size) * mean_stddev
        if h.is_cuda():
            mbatch_stddev = mbatch_stddev.cuda()
        h = torch.cat([h, mbatch_stddev], dim=1)
        return h

