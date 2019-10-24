"""
    The GAN image sample logger produces samples from G for every epoch with constant latent vectors and writes them
    to disk. It requires a GAN-based training loop.
"""
import math
import os
from argparse import Namespace
import torchvision.utils
import util.output
from trainloops.listeners.listener import Listener


class GanImageSampleLogger(Listener):
    def __init__(self, experiment_output_path, args: Namespace, n_images=16, pad_value=0):
        super().__init__()
        self.path = os.path.join(experiment_output_path, "imgs", "generator_samples")
        util.output.make_result_dirs(self.path)
        self.cuda = args.cuda
        self.n_images = n_images
        self.z = None
        self.pad_value = pad_value
        self.rows = int(math.ceil(math.sqrt(n_images)))

    def initialize(self):
        self.z = self.trainloop.generate_z_batch(self.n_images)

    def report(self, state_dict):
        epoch = state_dict["epoch"]
        G = state_dict["G"]

        images = G(self.z)
        torchvision.utils.save_image(
            images,
            os.path.join(self.path, "output-%06d.png"%epoch),
            nrow=self.rows,
            pad_value=self.pad_value
        )
