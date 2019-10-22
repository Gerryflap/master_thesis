import torch
import torch.nn.functional as F


def mish(x):
    # Mish activation https://arxiv.org/pdf/1908.08681v2.pdf
    return x * torch.tanh(F.softplus(x))
