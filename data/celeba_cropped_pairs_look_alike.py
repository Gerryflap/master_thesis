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


class CelebaCroppedPairsLookAlike(VisionDataset):
    cropped_base_folder = "celeba_cropped/img_align/"

    def __init__(self, split="train", transform=None, target_transform=None, download=False):
        super().__init__("data", transforms=None, transform=transform, target_transform=target_transform)

        if not os.path.isdir("data/celeba_cropped/"):
            # Initialize the dataset if it does not yet exist
            CelebaCropped(split, transform, target_transform, download)

        if split not in ["valid", "test", "better_valid"]:
            raise ValueError("This Dataset can only be used for evaluation (valid or test)!")

        # Load filenames and splits
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "better_valid": 3,
            "all": None,
        }
        split = split_map[split]

        if split == 1:
            with open("data/celeba_cropped/valid_pairs.json", "r") as f:
                self.pairs = json.load(f)
        elif split == 3:
            with open("data/celeba_cropped/better_pairs_1.json", "r") as f:
                self.pairs = json.load(f)
        else:
            with open("data/celeba_cropped/test_pairs.json", "r") as f:
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

if __name__ == "__main__":
    ds = CelebaCroppedPairsLookAlike(download=True)

