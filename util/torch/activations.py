import torch
import torch.nn.functional as F


def mish(x):
    # Mish activation https://arxiv.org/pdf/1908.08681v2.pdf
    return x * torch.tanh(F.softplus(x))


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