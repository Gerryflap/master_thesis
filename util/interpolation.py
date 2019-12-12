import numpy as np


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
