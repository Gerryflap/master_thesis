import torch


class UpscaleLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_channels=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.to_rgb = torch.nn.Conv2d(out_channels, n_channels, kernel_size=1)

    def forward(self, inp):
        x = torch.nn.functional.interpolate(inp, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        x = self.conv2(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        rgb = self.to_rgb(x)
        return x, rgb


class DeepGenerator(torch.nn.Module):
    def __init__(self, latent_size, h_size, start_resolution, n_upscales, n_channels=3):
        super().__init__()
        self.start_resolution = start_resolution
        self.n_upscales = n_upscales
        self.h_size = h_size
        self.n_channels = n_channels
        self.latent_size = latent_size

        self.lin = torch.nn.Linear(
            latent_size,
            int(h_size*(2**(n_upscales - 1)) * start_resolution**2)
        )
        self.first_rgb = torch.nn.Conv2d(h_size * 2 ** (n_upscales - 1), n_channels, kernel_size=1)

        upscale_layers = []
        for i in range(n_upscales):
            layer = UpscaleLayer(
                int(h_size * (2**(n_upscales - i - 1))),
                int(h_size * (2**(n_upscales - i - 2))),
                n_channels
            )
            upscale_layers.append(layer)
        self.upscale_layers = torch.nn.ModuleList(upscale_layers)

    def forward(self, z):
        x = self.lin(z)
        x = torch.nn.functional.leaky_relu(x, 0.02)
        x = x.view(
            -1,
            int(self.h_size * (2**(self.n_upscales - 1))),
            self.start_resolution,
            self.start_resolution
        )

        rgb_out = self.first_rgb(x)

        for layer in self.upscale_layers:
            x, rgb = layer(x)
            rgb_out = torch.nn.functional.upsample_bilinear(rgb_out, scale_factor=2)
            rgb_out += rgb

        return torch.sigmoid(rgb_out)


if __name__ == "__main__":
    G = DeepGenerator(256, 4, 6, 3)
    print(G(torch.normal(0, 1, (2, 256))).size())
