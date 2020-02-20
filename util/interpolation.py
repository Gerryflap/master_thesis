import numpy as np
import torch


def slerp(val, low, high):
    # https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
    # spherical linear interpolation, or slerp, does takes into account that the latent space is not a hypercube,
    #   but a hypersphere.
    # Assuming two latent vectors with an equal distance to the center,
    #   the linear interpolation between those will always be closer to the center of the hypersphere.
    # This interpolation method attempts to fix this
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def torch_slerp(val, low, high, dim=0):
    omega = torch.acos(((low/low.norm(dim=dim, keepdim=True)) * (high/high.norm(dim=dim, keepdim=True))).sum(dim=dim, keepdim=True))
    so = torch.sin(omega)
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega)/so * high


if __name__ == "__main__":
    z1, z2 = torch.normal(0, 1, (256,)), torch.normal(0, 1, (256,))

    np_interpolation = torch.from_numpy(slerp(0.5, z1.numpy(), z2.numpy()))
    torch_interpolation = torch_slerp(0.5, z1, z2)

    print(np_interpolation)
    print(torch_interpolation)

    assert (np_interpolation == torch_interpolation).all()