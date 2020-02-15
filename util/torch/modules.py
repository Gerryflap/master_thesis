import torch

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
        self.he_constant = torch.sqrt(2.0/float(in_channels))

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight/self.he_constant
        x = torch.nn.functional.conv2d(inp, weight, self.bias, self.stride, self.padding)
        return x
