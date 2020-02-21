"""
    This listener generates images with 3 rows:
        x1, x_morph, x2
    It can be used to evaluate morphing performance

"""
import math
import os
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
import torchvision.utils
import util.output
from trainloops.listeners.listener import Listener
from util.interpolation import torch_slerp


class MorphImageLogger(Listener):
    def __init__(self, experiment_output_path, validation_dataset, args: Namespace, n_images=16, pad_value=0, folder_name="morph_samples", eval_mode=True, slerp=False):
        super().__init__()
        self.path = os.path.join(experiment_output_path, "imgs", folder_name)
        util.output.make_result_dirs(self.path)
        self.cuda = args.cuda
        self.n_images = n_images
        self.x1 = None
        self.x2 = None
        self.pad_value = pad_value
        self.loader = DataLoader(validation_dataset, self.n_images, True)
        self.eval_mode = eval_mode
        self.slerp = slerp

    def initialize(self):
        self.x1, self.x2 = self.loader.__iter__().__next__()
        if self.cuda:
            self.x1 = self.x1.cuda()
            self.x2 = self.x2.cuda()

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

        if self.eval_mode:
            Gx.eval()
            Gz.eval()

        if self.slerp:
            z1 = Gz.encode(self.x1)
            z2 = Gz.encode(self.x2)
            z_morph = torch_slerp(0.5, z1, z2)
        else:
            z_morph = Gz.morph(self.x1, self.x2)

        x_morph = Gx(z_morph)

        if self.x1.detach().min().item() >= 0:
            range_  = (0, 1)
        else:
            range_ = (-1, 1)


        img = torchvision.utils.make_grid(
            torch.cat([self.x1, x_morph, self.x2], dim=0),
            nrow=self.n_images,
            pad_value=self.pad_value,
            range=range_,
            normalize=True
        )

        torchvision.utils.save_image(
            img,
            os.path.join(self.path, "output-%06d.png"%epoch),
            pad_value=self.pad_value,
        )

        if self.eval_mode:
            Gx.train()
            Gz.train()