import torch
from util.torch.activations import mish
from util.torch.initialization import weights_init


class VEEGANDiscriminator28(torch.nn.Module):
    def __init__(self, latent_size, h_size, fc_h_size=None, use_bn=False, use_mish=False, n_channels=1, dropout=0.0, use_logits=True):
        super().__init__()

        self.use_logits = use_logits
        self.n_channels = n_channels
        self.latent_size = latent_size

        if use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.dropout = dropout
        self.h_size = h_size
        if fc_h_size is None:
            self.fc_h_size = h_size*4
        else:
            self.fc_h_size = fc_h_size
        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=4,  stride=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=5, stride=2, bias=False)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, bias=False)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=4, stride=1, bias=False)

        self.use_bn = use_bn
        if use_bn:
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)

        if dropout != 0:
            self.dropout_layer = torch.nn.Dropout(dropout, False)

        self.lin_xz1 = torch.nn.Linear(h_size*4 + self.latent_size, self.fc_h_size, bias=True)
        self.lin_xz2 = torch.nn.Linear(self.fc_h_size, 1, bias=True)

    def init_weights(self):
        self.apply(weights_init)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x, z = inp

        h_x = self.compute_dx(x)

        prediction = self.compute_dxz(h_x, z)
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

        # Flatten to batch of vectors
        h_x = h.view(-1, self.h_size * 4)
        return h_x

    def compute_dxz(self, h_x, h_z):
        h = torch.cat((h_x, h_z), dim=1)

        if self.dropout != 0:
            h = self.dropout_layer(h)
        h = self.lin_xz1(h)
        h = self.activ(h)

        if self.dropout != 0:
            h = self.dropout_layer(h)
        h = self.lin_xz2(h)

        return h

if __name__ == "__main__":
    D = VEEGANDiscriminator28(128, 32, n_channels=3)
    print(D((torch.zeros((10, 3, 28, 28)), torch.zeros(10, 128))).size())