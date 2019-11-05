"""
    The GAN image sample logger produces samples from G for every epoch with constant latent vectors and writes them
    to disk. It requires a training loop that either has a G (generator) or a dec (decoder). It will also work for VAEs.
"""
import math
import os
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
import torchvision.utils
import util.output
from trainloops.listeners.listener import Listener


class AEImageSampleLogger(Listener):
    def __init__(self, experiment_output_path, validation_dataset, args: Namespace, n_images=16, pad_value=0):
        super().__init__()
        self.path = os.path.join(experiment_output_path, "imgs", "AE_samples")
        util.output.make_result_dirs(self.path)
        self.cuda = args.cuda
        self.n_images = n_images
        self.z = None
        self.pad_value = pad_value
        self.loader = DataLoader(validation_dataset, self.n_images, True)

    def initialize(self):
        self.z = self.trainloop.generate_z_batch(self.n_images)
        self.x = self.loader.__iter__().__next__()[0]
        if self.cuda:
            self.x = self.x.cuda()

        # Remove the loader since we got our test images
        del self.loader

    def report(self, state_dict):
        epoch = state_dict["epoch"]

        if "G" in state_dict["networks"]:
            Gx = state_dict["networks"]["G"]
        elif "dec" in state_dict["networks"]:
            Gx = state_dict["networks"]["dec"]
        elif "Gx" in state_dict["networks"]:
            Gx = state_dict["networks"]["Gx"]
        else:
            raise ValueError("Could not find a decoder-like network in the state dict!")

        if "enc" in state_dict["networks"]:
            Gz = state_dict["networks"]["enc"]
        elif "Gz" in state_dict["networks"]:
            Gz = state_dict["networks"]["Gz"]
        else:
            raise ValueError("Could not find a encoder-like network in the state dict!")

        generated_images = Gx(self.z)

        z_recon = Gz.encode(self.x)
        x_recon = Gx(z_recon)


        recon_grid = torchvision.utils.make_grid(
            torch.cat([self.x, x_recon], dim=0),
            nrow=self.n_images,
            pad_value=self.pad_value,
            range=(-1, 1),
            normalize=True
        )

        gen_grid = torchvision.utils.make_grid(
            generated_images,
            nrow=self.n_images,
            pad_value=self.pad_value,
            range=(-1, 1),
            normalize=True
        )

        img = torch.cat([recon_grid, gen_grid], 1)

        torchvision.utils.save_image(
            img,
            os.path.join(self.path, "output-%06d.png"%epoch),
            pad_value=self.pad_value,
        )
