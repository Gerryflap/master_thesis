import torch

from util.torch.initialization import weights_init


class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# Untested and unused
class Conv2dNormalizedLR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.he_constant = (2.0/float(in_channels*kernel_size*kernel_size))**0.5

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        x = torch.nn.functional.conv2d(inp, weight, self.bias, self.stride, self.padding)
        return x

    def reset_parameters(self):
        self.apply(weights_init)


class LinearNormalizedLR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias=True):
        super().__init__()
        self.he_constant = (2.0/float(in_channels))**0.5

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        x = torch.nn.functional.linear(inp, weight, self.bias)
        return x

    def reset_parameters(self):
        self.apply(weights_init)