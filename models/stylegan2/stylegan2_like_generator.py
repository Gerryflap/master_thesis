import torch

from util.torch.activations import LocalResponseNorm
from util.torch.initialization import weights_init


class UpscaleLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_channels=3, bn=False, lrn=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.to_rgb = torch.nn.Conv2d(out_channels, n_channels, kernel_size=1)

        self.norm = bn or lrn
        if bn:
            self.n_1 = torch.nn.BatchNorm2d(out_channels)
            self.n_2 = torch.nn.BatchNorm2d(out_channels)
        elif lrn:
            self.n_1 = LocalResponseNorm()
            self.n_2 = LocalResponseNorm()


    def forward(self, inp):
        x = torch.nn.functional.interpolate(inp, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)
        if self.norm:
            x = self.n_1(x)

        x = self.conv2(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)
        if self.norm:
            x = self.n_2(x)

        rgb = self.to_rgb(x)
        return x, rgb


class DeepGenerator(torch.nn.Module):
    def __init__(self, latent_size, h_size, start_resolution, n_upscales, n_channels=3, bn=False, lrn=False):
        super().__init__()
        self.start_resolution = start_resolution
        self.n_upscales = n_upscales
        self.h_size = h_size
        self.n_channels = n_channels
        self.latent_size = latent_size

        self.lin = torch.nn.Linear(
            latent_size,
            int(h_size*(2**n_upscales) * start_resolution**2)
        )
        self.first_rgb = torch.nn.Conv2d(h_size * 2 ** (n_upscales ), n_channels, kernel_size=1)

        self.norm = bn or lrn
        if bn:
            self.n_1 = torch.nn.BatchNorm2d(h_size*(2**n_upscales))
        if lrn:
            self.n_1 = LocalResponseNorm()

        upscale_layers = []
        for i in range(n_upscales):
            layer = UpscaleLayer(
                int(h_size * (2**(n_upscales - i))),
                int(h_size * (2**(n_upscales - i - 1))),
                n_channels,
                bn,
                lrn
            )
            upscale_layers.append(layer)
        self.upscale_layers = torch.nn.ModuleList(upscale_layers)

    def forward(self, z):
        x = self.lin(z)
        x = torch.nn.functional.leaky_relu(x, 0.02)
        x = x.view(
            -1,
            int(self.h_size * (2**(self.n_upscales))),
            self.start_resolution,
            self.start_resolution
        )

        x = self.n_1(x)

        rgb_out = self.first_rgb(x)

        for layer in self.upscale_layers:
            x, rgb = layer(x)
            rgb_out = torch.nn.functional.interpolate(rgb_out, scale_factor=2, mode='bilinear', align_corners=True)
            rgb_out += rgb

        return torch.sigmoid(rgb_out)

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    G = DeepGenerator(256, 4, 6, 3)
    print(G(torch.normal(0, 1, (2, 256))).size())
