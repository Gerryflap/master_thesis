import torch

from util.torch.initialization import weights_init

class DownscaleLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, z_channels, bn=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels + z_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.lower_channels = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.bn = bn
        if bn:
            self.bn_1 = torch.nn.BatchNorm2d(out_channels)
            self.bn_2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, inp):
        inp, z_in = inp

        carry = torch.nn.functional.interpolate(inp, scale_factor=0.5, mode='bilinear', align_corners=True)
        carry = self.lower_channels(carry)

        x = torch.cat([inp, z_in], dim=1)

        x = self.conv1(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        if self.bn:
            x = self.bn_1(x)

        x = self.conv2(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        if self.bn:
            x = self.bn_2(x)

        return carry + x


class DeepAltAliDiscriminator(torch.nn.Module):
    def __init__(self, latent_size, h_size, input_resolution, n_downscales, n_channels=3, bn=False, z_channels=16):
        super().__init__()
        self.input_resolution = input_resolution
        self.n_downscales = n_downscales
        self.conv_out_res = int(input_resolution * 0.5**n_downscales)
        self.latent_size = latent_size
        self.h_size = h_size
        self.n_channels = n_channels
        self.bn = bn
        self.z_channels = z_channels

        self.from_rgb = torch.nn.Conv2d(n_channels, h_size, kernel_size=1)

        downscale_layers = []
        for i in range(n_downscales):
            layer = DownscaleLayer(
                int(h_size * (2**i)),
                int(h_size * (2**(i + 1))),
                z_channels,
                bn and (i != 0)
            )
            downscale_layers.append(layer)
        self.downscale_layers = torch.nn.ModuleList(downscale_layers)

        z_channel_layers = []
        for i in range(n_downscales):
            layer = torch.nn.Linear(latent_size, z_channels)
            z_channel_layers.append(layer)
        self.z_channel_layers = torch.nn.ModuleList(z_channel_layers)

        self.x_fc = torch.nn.Linear(
            int(h_size*(2**(n_downscales)) * self.conv_out_res**2),
            int(h_size*(2**(n_downscales + 1)))
        )

        self.z_fc = torch.nn.Linear(
            latent_size,
            int(h_size*(2**(n_downscales + 1)))
        )

        self.xz_fc = torch.nn.Linear(
            int(h_size*(2**(n_downscales + 2))),
            1
        )


    def forward(self, inp):
        x, z = inp

        x = self.from_rgb(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        for layer, z_layer in zip(self.downscale_layers, self.z_channel_layers):
            z_channels = self.activ(z_layer(z))
            z_channels = z_channels.view(-1, self.z_channels, 1, 1)
            z_channels = z_channels.repeat(1, 1, x.size(2), x.size(3))
            x = layer((x, z_channels))

        x = x.view(-1, int(self.h_size * (2 ** self.n_downscales)) * self.conv_out_res * self.conv_out_res)
        hx = self.x_fc(x)
        hx = self.activ(hx)

        hz = self.z_fc(z)
        hz = self.activ(hz)

        hxz = torch.cat([hx, hz], dim=1)
        out = self.xz_fc(hxz)
        return out

    @staticmethod
    def activ(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    D = DeepAltAliDiscriminator(128, 4, 48, 3)
    print(D((torch.normal(0, 1, (2, 3, 48, 48)), torch.normal(0, 1, (2, 128)))).size())
