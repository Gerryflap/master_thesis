"""
    Models a Gz/Encoder that is able to generate morphs.
    This allows for a single method that captures all morphing methods proposed in RT, apart from gradient descend.
    This method is meant to be overridden by some encoders but the default implementation takes the mean.
"""
import torch


class MorphingEncoder(torch.nn.Module):
    def morph(self, x1, x2, use_mean=False):
        """
        Morphs the images in x1 with the images in x2 and returns the outcome latent representation.
        :param x1: A batch of images of the first identities to be morphed
        :param x2: A batch of images to morph with the x1 images
        :param use_mean: use z_mean instead of sampling from q(z|x)
        :return: A batch of morphed z values. These will have to go through the decoder/Gx in order to decode.
        """
        z1, z2 = self.encode(x1, use_mean=use_mean), self.encode(x2, use_mean=use_mean)
        z = 0.5*(z1 + z2)
        return z

    def encode(self, x, use_mean=False):
        """
        Encodes x to a latent vector. This method exists to unify the return values.
        Different models might return more values when called directly.

        The default implementation assumes a VAE like encoder that returns a 3-tuple
        where the first element is a sample. If this is not the case, the method should be overridden.
        :param x: A batch of images
        :return: A list of latent representations of these images in x
        """

        z, zm, _ = self(x)
        if not use_mean:
            return z
        else:
            return zm
