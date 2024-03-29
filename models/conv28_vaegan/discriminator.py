import torch
from util.torch.initialization import weights_init


class VAEGANDiscriminator28(torch.nn.Module):
    def __init__(self, h_size, use_bn=False, n_channels=1, dropout=0.0):
        super().__init__()
        self.n_channels = n_channels

        self.activ = torch.nn.functional.relu

        self.dropout = dropout
        self.h_size = h_size
        self.conv_1 = torch.nn.Conv2d(n_channels, h_size // 2, kernel_size=5, stride=1, padding=2)
        self.conv_2 = torch.nn.Conv2d(h_size // 2, h_size * 2, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2, padding=2, bias=False)

        self.use_bn = use_bn
        if use_bn:
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_fc = torch.nn.BatchNorm1d(self.h_size * 8)

        if dropout != 0:
            self.dropout_layer = torch.nn.Dropout(dropout, True)
            self.dropout_conv_layer = torch.nn.Dropout2d(dropout, True)

        self.lin_1 = torch.nn.Linear(7 * 7 * h_size * 4, h_size * 8, bias=False)
        self.lin_2 = torch.nn.Linear(h_size * 8, 1)

    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def forward(self, inp):
        x = self.conv_1(inp)
        if self.dropout != 0:
            x = self.dropout_conv_layer(x)
        x = self.activ(x)

        x = self.conv_2(x)
        if self.dropout != 0:
            x = self.dropout_conv_layer(x)
        if self.use_bn:
            x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        dis_l = x
        if self.dropout != 0:
            x = self.dropout_conv_layer(x)
        if self.use_bn:
            x = self.bn_3(x)
        x = self.activ(x)

        # Flatten to vector
        x = x.view(-1, 7 * 7 * self.h_size * 4)

        if self.dropout != 0:
            x = self.dropout_layer(x)

        x = self.lin_1(x)
        if self.use_bn:
            x = self.bn_fc(x)
        x = self.activ(x)

        if self.dropout != 0:
            x = self.dropout_layer(x)

        x = self.lin_2(x)
        # For numerical stability, logits are used as output instead.
        # This means that the sigmoid function "is embedded" in the loss function directly
        return x, dis_l

    def init_weights(self):
        self.apply(weights_init)
