import torch

from util.torch.initialization import weights_init
from models.stylegan2.stylegan2_like_discriminator import DownscaleLayer
from util.torch.modules import Conv2dNormalizedLR, LinearNormalizedLR


class DeepAliDiscriminator(torch.nn.Module):
    def __init__(self, latent_size, h_size, input_resolution, n_downscales, n_channels=3, bn=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.n_downscales = n_downscales
        self.conv_out_res = int(input_resolution * 0.5**n_downscales)
        self.latent_size = latent_size
        self.h_size = h_size
        self.n_channels = n_channels
        self.bn = bn

        self.from_rgb = Conv2dNormalizedLR(n_channels, h_size, kernel_size=1)

        downscale_layers = []
        for i in range(n_downscales):
            layer = DownscaleLayer(
                int(h_size * (2**i)),
                int(h_size * (2**(i + 1))),
                bn and (i != 0)
            )
            downscale_layers.append(layer)
        self.downscale_layers = torch.nn.ModuleList(downscale_layers)

        self.x_fc = LinearNormalizedLR(
            int(h_size*(2**(n_downscales)) * self.conv_out_res**2),
            h_size * (2 ** (n_downscales + 1))
        )

        self.fc_h_size = h_size * (2 ** (n_downscales + 1))

        self.lin_z1 = LinearNormalizedLR(latent_size, self.fc_h_size, bias=False)
        self.lin_z2 = LinearNormalizedLR(self.fc_h_size, self.fc_h_size, bias=False)

        self.lin_xz1 = LinearNormalizedLR(self.fc_h_size*2, self.fc_h_size*2, bias=True)
        self.lin_xz2 = LinearNormalizedLR(self.fc_h_size*2, self.fc_h_size*2, bias=True)
        self.lin_xz3 = LinearNormalizedLR(self.fc_h_size*2, 1, bias=True)

    def compute_dx(self, x):
        x = self.from_rgb(x)
        x = torch.nn.functional.leaky_relu(x, 0.02)

        for layer in self.downscale_layers:
           x = layer(x)

        dis_l = x
        x = x.view(-1, int(self.h_size * (2 ** self.n_downscales)) * self.conv_out_res * self.conv_out_res)
        x = self.x_fc(x)

        return x, dis_l

    def compute_dz(self, z):
        h_z = self.lin_z1(z)
        h_z = self.activ(h_z)

        h_z = self.lin_z2(h_z)
        h_z = self.activ(h_z)

        return h_z

    def compute_dxz(self, h_x, h_z):
        h = torch.cat((h_x, h_z), dim=1)

        h = self.lin_xz1(h)
        h = self.activ(h)

        h = self.lin_xz2(h)
        h = self.activ(h)

        h = self.lin_xz3(h)
        return h


    def forward(self, inp):
        x, z = inp

        h_x, dis_l = self.compute_dx(x)
        h_z = self.compute_dz(z)

        prediction = self.compute_dxz(h_x, h_z)
        # There is no sigmoid applied to the output here.
        # This is done in the loss function for improved numerical stability
        # While this does make the code more confusing, it drastically improves the stability in practice.
        return prediction

    @staticmethod
    def activ(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

    @staticmethod
    def sample(means, vars):
        stds = torch.exp(0.5 * vars)
        eps = torch.randn_like(stds)
        return means + eps * stds

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    D = DeepAliDiscriminator(128, 4, 48, 3)
    print(D((torch.normal(0, 1, (2, 3, 48, 48)), torch.normal(0, 1, (2, 128)))).size())
