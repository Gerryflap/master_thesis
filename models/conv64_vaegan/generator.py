import torch

from util.torch.activations import mish
from util.torch.initialization import weights_init


class VAEGANGenerator64(torch.nn.Module):
    def __init__(self, latent_size, h_size, n_channels=1):
        super().__init__()

        self.n_channels = n_channels

        self.latent_size = latent_size
        self.h_size = h_size

        # VAE/GAN uses some layers with 32 filters.
        # To keep the h_size roughly similar in overall network size between papers, this is taken to be h_size = 64.
        # Therefore some layers will have h_size//2 channels
        assert h_size % 2 == 0

        self.activ = mish

        self.conv_1 = torch.nn.ConvTranspose2d(self.latent_size, self.h_size * 4, 8, bias=False)
        self.conv_2 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size * 4, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1)
        self.conv_3 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size*2, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1)
        self.conv_4 = torch.nn.ConvTranspose2d(self.h_size * 2, self.h_size//2, kernel_size=5, stride=2, bias=False,padding=2, output_padding=1)
        self.conv_5 = torch.nn.ConvTranspose2d(self.h_size//2, n_channels, kernel_size=5, stride=1, bias=False, padding=2)

        self.output_bias = torch.nn.Parameter(torch.zeros((3, 64, 64)), requires_grad=True)

        self.bn_1 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_4 = torch.nn.BatchNorm2d(self.h_size//2)




    def forward(self, inp):
        x = inp.view(-1, self.latent_size, 1, 1)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activ(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.activ(x)

        x = self.conv_5(x)
        # TODO: This check allows for older models to run without the output bias. It will be removed in the future
        if hasattr(self, "output_bias"):
            x = x + self.output_bias
        x = torch.tanh(x)

        return x

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    gen = VAEGANGenerator64(10, 32)
    z = torch.zeros((1,10))
    x_out = gen(z)
    res = int(x_out.size()[3])

    print("Output image resolution: ", res)