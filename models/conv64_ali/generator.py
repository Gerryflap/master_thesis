import torch
from util.torch.activations import mish, LocalResponseNorm
from util.torch.initialization import weights_init


class Generator64(torch.nn.Module):
    def __init__(self, latent_size, h_size, use_mish=False, n_channels=3, sigmoid_out=False, use_lr_norm=False):
        super().__init__()

        self.n_channels = n_channels

        self.latent_size = latent_size
        self.h_size = h_size
        if use_mish:

            self.activ = mish
        else:
            self.activ = self.leaky_relu
        self.sigmoid_out = sigmoid_out

        self.conv_1 = torch.nn.ConvTranspose2d(self.latent_size, self.h_size * 8, 4, bias=False)
        self.conv_2 = torch.nn.ConvTranspose2d(self.h_size * 8, self.h_size * 4, kernel_size=7, stride=2, bias=False)
        self.conv_3 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size*4, kernel_size=5, stride=2, bias=False)
        self.conv_4 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size*2, kernel_size=7, stride=2, bias=False)
        self.conv_5 = torch.nn.ConvTranspose2d(self.h_size*2, self.h_size*1, kernel_size=2, stride=1, bias=False)
        self.conv_6 = torch.nn.Conv2d(self.h_size, self.n_channels, kernel_size=1, stride=1, bias=False)

        self.output_bias = torch.nn.Parameter(torch.zeros((3, 64, 64)), requires_grad=True)

        if not use_lr_norm:
            self.bn_1 = torch.nn.BatchNorm2d(self.h_size * 8)
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 2)
            self.bn_5 = torch.nn.BatchNorm2d(self.h_size)
        else:
            self.bn_1 = LocalResponseNorm()
            self.bn_2 = LocalResponseNorm()
            self.bn_3 = LocalResponseNorm()
            self.bn_4 = LocalResponseNorm()
            self.bn_5 = LocalResponseNorm()



    @staticmethod
    def leaky_relu(x):
        return torch.nn.functional.leaky_relu(x, 0.02)

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
        x = self.bn_5(x)
        x = self.activ(x)

        x = self.conv_6(x)
        # TODO: This check allows for older models to run without the output bias. It will be removed in the future
        if hasattr(self, "output_bias"):
            x = x + self.output_bias
        if self.sigmoid_out:
            x = torch.sigmoid(x)
        else:
            x = torch.tanh(x)
        return x

    def init_weights(self):
        self.apply(weights_init)


if __name__ == "__main__":
    G = Generator64(256, 64)
    G(torch.normal(0, 1, (1, 256)))
    print(list([param.size() for param in G.parameters()]))
    
