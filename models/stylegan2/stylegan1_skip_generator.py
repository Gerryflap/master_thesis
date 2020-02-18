"""
    This generator uses stylegan 1 style blocks, but uses a skip connection design from stylegan2.
    Therefore the Style blocks also output an RGB image"
"""
import torch

from util.torch.activations import adaptive_instance_normalization
from util.torch.initialization import weights_init
from util.torch.modules import LinearNormalizedLR, Conv2dNormalizedLR


class StyleUpscaleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_size, n_channels=3, upsample=True):
        super().__init__()
        self.affine1 = LinearNormalizedLR(w_size, out_channels*2, bias=False)
        self.affine2 = LinearNormalizedLR(w_size, out_channels*2, bias=False)

        self.conv1 = Conv2dNormalizedLR(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dNormalizedLR(out_channels, out_channels, kernel_size=3, padding=1)
        self.to_rgb = Conv2dNormalizedLR(out_channels, n_channels, kernel_size=1)

        self.upsample = upsample
        self.out_channels = out_channels

    def forward(self, inp):
        x, w = inp
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')

        y1 = self.affine1(w)
        y2 = self.affine2(w)

        x = self.conv1(x)
        x = adaptive_instance_normalization(x, y1[:, :self.out_channels], y1[:, self.out_channels:])
        x = torch.nn.functional.leaky_relu(x, 0.02)

        x = self.conv2(x)
        x = adaptive_instance_normalization(x, y2[:, :self.out_channels], y2[:, self.out_channels:])
        x = torch.nn.functional.leaky_relu(x, 0.02)

        rgb = self.to_rgb(x)
        return x, rgb


class StyleSynthesisNetwork(torch.nn.Module):
    def __init__(self, latent_size, h_size, start_resolution, n_upscales, n_channels=3):
        super().__init__()
        self.start_resolution = start_resolution
        self.n_upscales = n_upscales
        self.h_size = h_size
        self.n_channels = n_channels
        self.latent_size = latent_size

        self.constant_input = torch.nn.Parameter(
            torch.normal(0, 1, (1, h_size * (2 ** n_upscales), start_resolution, start_resolution)),
            requires_grad=True
        )
        self.first_layer = StyleUpscaleBlock(h_size * (2 ** n_upscales), h_size * (2 ** n_upscales),
                                             latent_size, upsample=False)

        upscale_layers = []
        for i in range(n_upscales):
            layer = StyleUpscaleBlock(
                int(h_size * (2**(n_upscales - i))),
                int(h_size * (2**(n_upscales - i - 1))),
                latent_size,
                n_channels,
            )
            upscale_layers.append(layer)
        self.upscale_layers = torch.nn.ModuleList(upscale_layers)

    def forward(self, w):
        const = self.constant_input

        # Repeat along batch dim
        const = const.repeat(w.size(0), 1, 1, 1)

        x, rgb_out = self.first_layer((const, w))

        for layer in self.upscale_layers:
            x, rgb = layer((x, w))
            rgb_out = torch.nn.functional.interpolate(rgb_out, scale_factor=2, mode='bilinear', align_corners=False)
            rgb_out += rgb

        return torch.sigmoid(rgb_out)

    def init_weights(self):
        self.apply(weights_init)


class StyleMappingNetwork(torch.nn.Module):
    def __init__(self, latent_size, depth=8):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.append(LinearNormalizedLR(latent_size, latent_size))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.02)
        x = self.layers[-1](x)
        return x

    def init_weights(self):
        self.apply(weights_init)


class StyleGenerator(torch.nn.Module):
    """
        The StyleGenerator combines the StyleSynthesis and StyleMapping networks.
         Use the provided methods for acquiring the parameters of both separately
          and apply a lower learning rate to the mapping network
    """
    def __init__(self, latent_size, h_size, start_resolution, n_upscales, n_channels=3, mapping_depth=8):
        super().__init__()
        self.latent_size = latent_size
        self.mapping_net = StyleMappingNetwork(latent_size, mapping_depth)
        self.synthesis_net = StyleSynthesisNetwork(latent_size, h_size, start_resolution, n_upscales, n_channels)

    def get_mapping_parameters(self):
        return self.mapping_net.parameters()

    def get_synthesis_parameters(self):
        return self.synthesis_net.parameters()

    def forward(self, inp):
        w = self.mapping_net(inp)
        return self.synthesis_net(w)

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    G = StyleGenerator(256, 4, 6, 3)
    print(G(torch.normal(0, 1, (2, 256))).size())
