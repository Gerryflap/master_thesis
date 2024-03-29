"""
    This is an attempt at implementing pre-activation residual layers.
    They are not guaranteed to perfectly copy the procedure described in https://arxiv.org/abs/1603.05027
        but are inspired by it
"""
import torch

from util.torch.activations import mish


class ResidualConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, downscale=False, dropout_rate=0.0, bias=None):
        super().__init__()

        if bias is None:
            bias = not bn

        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2 if downscale else 1, padding=1, bias=bias)
        self.conv_2 = torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=bias)


        self.bn = bn
        self.downscale = downscale

        if bn:
            self.bn_1 = torch.nn.BatchNorm2d(in_channels)
            self.bn_2 = torch.nn.BatchNorm2d(out_channels)

        if downscale:
            self.conv_down = torch.nn.Conv2d(in_channels, out_channels, 2, stride=2)

        self.dropout = None
        if dropout_rate != 0:
            self.dropout = torch.nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x_carry = x

        if self.downscale:
            x_carry = self.conv_down(x_carry)

        if self.dropout is not None:
            x = self.dropout(x)
        if self.bn:
            x = self.bn_1(x)
        x = mish(x)

        x = self.conv_1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.bn:
            x = self.bn_2(x)
        x = mish(x)
        x = self.conv_2(x)

        return x_carry + x


class ResidualConvolutionTransposeLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, upscale=False, dropout_rate=0.0, bias=None):
        super().__init__()

        if bias is None:
            bias = not bn

        self.conv_1 = torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2 if upscale else 1, padding=1, output_padding=1 if upscale else 0, bias=bias)
        self.conv_2 = torch.nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1, bias=bias)

        self.bn = bn
        self.upscale = upscale

        if bn:
            self.bn_1 = torch.nn.BatchNorm2d(in_channels)
            self.bn_2 = torch.nn.BatchNorm2d(out_channels)

        if upscale:
            self.conv_up = self.conv_down = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.dropout = None
        if dropout_rate != 0:
            self.dropout = torch.nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x_carry = x

        if self.upscale:
            x_carry = self.conv_up(x_carry)

        if self.dropout is not None:
            x = self.dropout(x)
        if self.bn:
            x = self.bn_1(x)
        x = mish(x)
        x = self.conv_1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.bn:
            x = self.bn_2(x)
        x = mish(x)
        x = self.conv_2(x)

        return x_carry + x


if __name__ == "__main__":
    m1 = torch.nn.Sequential(
        ResidualConvolutionLayer(4, 4),
        ResidualConvolutionLayer(4, 8, downscale=True),
    )

    m2 = torch.nn.Sequential(
        ResidualConvolutionTransposeLayer(8, 4, upscale=True),
        ResidualConvolutionLayer(4, 4),
    )

    inp = torch.normal(0, 1, (10, 4, 6, 6))

    hidden = m1(inp)
    print(hidden.size())
    out = m2(hidden)
    print(hidden.size())
