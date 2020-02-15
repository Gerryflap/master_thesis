import torch

from util.torch.initialization import weights_init
from models.stylegan2.stylegan2_like_discriminator import DownscaleLayer


class DeepAltAliDiscriminator(torch.nn.Module):
    def __init__(self, latent_size, h_size, input_resolution, n_downscales, n_channels=3, bn=False, z_channels=4):
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
                int(h_size * (2**i)) + z_channels,
                int(h_size * (2**(i + 1))),
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
            x = torch.cat([z_channels, x], dim=1)
            x = layer(x)

        x = x.view(-1, int(self.h_size * (2 ** self.n_downscales)) * self.conv_out_res * self.conv_out_res)
        x = self.x_fc(x)
        return x

    @staticmethod
    def activ(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    D = DeepAltAliDiscriminator(128, 4, 48, 3)
    print(D((torch.normal(0, 1, (2, 3, 48, 48)), torch.normal(0, 1, (2, 128)))).size())
