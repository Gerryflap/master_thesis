import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.initialization import weights_init
from models.stylegan2.stylegan2_like_discriminator import DownscaleLayer


class DeepEncoder(MorphingEncoder):
    def __init__(self, latent_size, h_size, input_resolution, n_downscales, n_channels=3, bn=False, lrn=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.n_downscales = n_downscales
        self.conv_out_res = int(input_resolution * 0.5**n_downscales)
        self.latent_size = latent_size
        self.h_size = h_size
        self.n_channels = n_channels
        self.bn = bn
        self.lrn = lrn

        self.from_rgb = torch.nn.Conv2d(n_channels, h_size, kernel_size=1)

        downscale_layers = []
        for i in range(n_downscales):
            layer = DownscaleLayer(
                int(h_size * (2**i)),
                int(h_size * (2**(i + 1))),
                bn and (i != 0),
                lrn and (i != 0)
            )
            downscale_layers.append(layer)
        self.downscale_layers = torch.nn.ModuleList(downscale_layers)

        self.mean_fc = torch.nn.Linear(
            int(h_size*(2**(n_downscales)) * self.conv_out_res**2),
            latent_size
        )

        self.logvar_fc = torch.nn.Linear(
            int(h_size*(2**(n_downscales)) * self.conv_out_res**2),
            latent_size
        )

    def forward(self, x):
        x = self.from_rgb(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        for layer in self.downscale_layers:
            x = layer(x)

        x = x.view(-1, int(self.h_size*(2**self.n_downscales)) * self.conv_out_res * self.conv_out_res)

        means = self.mean_fc(x)
        log_vars = self.logvar_fc(x)
        log_vars = -torch.nn.functional.softplus(log_vars)
        return self.sample(means, log_vars), means, log_vars

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    E = DeepEncoder(128, 4, 48, 3)
    print(E(torch.normal(0, 1, (2, 3, 48, 48)))[0].size())
