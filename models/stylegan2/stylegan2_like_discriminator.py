import torch

from util.torch.activations import LocalResponseNorm
from util.torch.initialization import weights_init
from util.torch.modules import Conv2dNormalizedLR, LinearNormalizedLR


class DownscaleLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, lrn=False):
        super().__init__()
        self.conv1 = Conv2dNormalizedLR(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv2dNormalizedLR(out_channels, out_channels, kernel_size=3, padding=1)
        self.lower_channels = Conv2dNormalizedLR(in_channels, out_channels, kernel_size=1)

        self.norm = bn or lrn
        if bn:
            self.n_1 = torch.nn.BatchNorm2d(out_channels)
            self.n_2 = torch.nn.BatchNorm2d(out_channels)
        elif lrn:
            self.n_1 = LocalResponseNorm()
            self.n_2 = LocalResponseNorm()


    def forward(self, inp):
        carry = torch.nn.functional.interpolate(inp, scale_factor=0.5, mode='bilinear', align_corners=True)
        carry = self.lower_channels(carry)

        x = self.conv1(inp)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        if self.norm:
            x = self.n_1(x)

        x = self.conv2(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        if self.norm:
            x = self.n_2(x)

        return carry + x


class DeepDiscriminator(torch.nn.Module):
    def __init__(self, h_size, input_resolution, n_downscales, n_channels=3, bn=False, lrn=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.n_downscales = n_downscales
        self.conv_out_res = int(input_resolution * 0.5**n_downscales)
        self.h_size = h_size
        self.n_channels = n_channels
        self.from_rgb = Conv2dNormalizedLR(n_channels, h_size, kernel_size=1)

        downscale_layers = []
        for i in range(n_downscales):
            layer = DownscaleLayer(
                int(h_size * (2**i)),
                int(h_size * (2**(i + 1))),
                bn and (i != 0),
                lrn and (i != 0),
            )
            downscale_layers.append(layer)
        self.downscale_layers = torch.nn.ModuleList(downscale_layers)

        self.out = LinearNormalizedLR(
            int(h_size*(2**(n_downscales)) * self.conv_out_res**2),
            1
        )

    def forward(self, x):
        x = self.compute_disl(x)

        x = x.view(-1, int(self.h_size*(2**self.n_downscales)) * self.conv_out_res * self.conv_out_res)
        x = self.out(x)

        return x

    def compute_disl(self, x):
        x = self.from_rgb(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        for layer in self.downscale_layers:
            x = layer(x)
        return x


    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    D = DeepDiscriminator(4, 48, 3)
    print(D(torch.normal(0, 1, (2, 3, 48, 48))).size())
