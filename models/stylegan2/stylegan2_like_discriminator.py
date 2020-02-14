import torch


class DownscaleLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.lower_channels = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inp):
        carry = torch.nn.functional.interpolate(inp, scale_factor=0.5, mode='bilinear', align_corners=True)
        carry = self.lower_channels(carry)

        x = self.conv1(inp)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        x = self.conv2(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        return carry + x


class DeepDiscriminator(torch.nn.Module):
    def __init__(self, h_size, input_resolution, n_downscales, n_channels=3):
        super().__init__()
        self.input_resolution = input_resolution
        self.n_downscales = n_downscales
        self.conv_out_res = int(input_resolution * 0.5**n_downscales)
        print(self.conv_out_res)
        self.h_size = h_size
        self.n_channels = n_channels

        self.from_rgb = torch.nn.Conv2d(n_channels, h_size, kernel_size=1)

        downscale_layers = []
        for i in range(n_downscales):
            layer = DownscaleLayer(
                int(h_size * (2**i)),
                int(h_size * (2**(i + 1)))
            )
            downscale_layers.append(layer)
        self.downscale_layers = torch.nn.ModuleList(downscale_layers)

        self.out = torch.nn.Linear(
            int(h_size*(2**(n_downscales)) * self.conv_out_res**2),
            1
        )

    def forward(self, x):
        x = self.from_rgb(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        for layer in self.downscale_layers:
            x = layer(x)

        x = x.view(-1, int(self.h_size*(2**self.n_downscales)) * self.conv_out_res * self.conv_out_res)
        x = self.out(x)

        return x


if __name__ == "__main__":
    D = DeepDiscriminator(4, 48, 3)
    print(D(torch.normal(0, 1, (2, 3, 48, 48))).size())
