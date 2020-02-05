import torch
import torch.nn.functional as F


def mish(x):
    # Mish activation https://arxiv.org/pdf/1908.08681v2.pdf
    return x * torch.tanh(F.softplus(x))


class Mish(torch.nn.Module):
    def forward(self, x):
        return mish(x)

class FilterResponseNormalization2d(torch.nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = torch.zeros((1, channels, 1, 1), requires_grad=True)
        self.gamma = torch.ones((1, channels, 1, 1), requires_grad=True)
        self.eps = torch.ones((1,), requires_grad=True)*eps

    def forward(self, x):
        nu2 = torch.mean(x.pow(2), (2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return self.gamma * x + self.beta


class FilterResponseNormalization(torch.nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = torch.zeros((1, channels), requires_grad=True)
        self.gamma = torch.ones((1, channels), requires_grad=True)
        self.eps = torch.zeros((1,), requires_grad=False)
        self.eps.fill_(eps)
        self.eps.requires_grad = True

    def forward(self, x):
        nu2 = x.pow(2)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return self.gamma * x + self.beta


class TLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = torch.zeros((1,), requires_grad=True)

    def forward(self, x):
        return torch.max(x, self.tau)


def local_response_normalization(x, eps=1e-8):
    """
    Implements the variant of LRN used in ProGAN https://arxiv.org/pdf/1710.10196.pdf
    :param eps: Epsilon is a small number added to the divisor to avoid division by zero
    :param x: Output of convolutional layer (or any other tensor with channels on axis 1)
    :return: Normalized x
    """
    divisor = (torch.pow(x, 2).mean(dim=1, keepdim=True) + eps).sqrt()
    b = x/divisor
    return b


class LocalResponseNorm(torch.nn.Module):
    def __init__(self, eps=1e-8):
        """
        Implements the variant of LRN used in ProGAN https://arxiv.org/pdf/1710.10196.pdf
        :param eps: Epsilon is a small number added to the divisor to avoid division by zero
        """
        super().__init__()
        self.eps = eps

    def forward(self, inp):
        return local_response_normalization(inp, self.eps)


def map_to_hypersphere(x, epsilon=1e-10):
    distances = torch.sqrt(torch.sum(x*x, dim=1, keepdim=True))
    return x/(distances + epsilon)