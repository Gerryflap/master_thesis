import torch
from util.torch.activations import mish
from util.torch.initialization import weights_init


class Discriminator28(torch.nn.Module):
    def __init__(self, h_size, use_bn=False, use_mish=False, n_channels=1, dropout=0.0, use_logits=True):
        super().__init__()

        self.use_logits = use_logits
        self.n_channels = n_channels

        if use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.dropout = dropout
        self.h_size = h_size
        self.conv_1 = torch.nn.Conv2d(n_channels, h_size, kernel_size=4,  stride=1)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=5, stride=2)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=4, stride=1)

        self.use_bn = use_bn
        if use_bn:
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2, 0.8)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4, 0.8)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4, 0.8)

        if dropout != 0:
            self.dropout_layer = torch.nn.Dropout(dropout, True)

        self.lin_1 = torch.nn.Linear(h_size * 4, 1)

        # Initialize weights
        self.apply(weights_init)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.activ(x)

        x = self.conv_2(x)
        if self.use_bn:
            x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        if self.use_bn:
            x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        if self.use_bn:
            x = self.bn_4(x)
        x = self.activ(x)

        # Flatten to vector
        x = x.view(-1, self.h_size * 4)

        if self.dropout != 0:
            x = self.dropout_layer(x)

        x = self.lin_1(x)
        if not self.use_logits:
            x = torch.sigmoid(x)
        return x