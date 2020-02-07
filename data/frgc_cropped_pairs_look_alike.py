"""
    Uses the pairs defined by data/split_generation/pair_generator.py output pairs of images in the following format:
    ((morph_inputs, comparison_images), (id_1, id_2))
    where both morph_inputs and comparison_images are tuples of images of (person_1, person_2).
"""
import random
import os
import json

import PIL.Image
from torchvision.datasets import VisionDataset
from data.celeba_cropped import CelebaCropped

from shutil import copyfile

assert os.path.isdir("data")


class FRGCPairsLookAlike(VisionDataset):
    cropped_base_folder = "frgc/cropped/"

    def __init__(self, transform=None, target_transform=None):
        super().__init__("data", transforms=None, transform=transform, target_transform=target_transform)

        with open("data/frgc/pairs.json", "r") as f:
            self.pairs = json.load(f)

    def __getitem__(self, index):
        pair = self.pairs[index]
        p_1 = pair[0]
        p_2 = pair[1]

        morph_inputs = self.load_image(p_1["morph_in"]), self.load_image(p_2["morph_in"])
        comparison_images = self.load_image(p_1["comparison_pic"]), self.load_image(p_2["comparison_pic"])
        idents = p_1["ident"], p_2["ident"]

        return (morph_inputs, comparison_images), idents

    def load_image(self, name):
        X = PIL.Image.open(os.path.join(self.root, self.cropped_base_folder, name))

        if self.transform is not None:
            X = self.transform(X)

        return X

    def __len__(self):
        return len(self.pairs)


