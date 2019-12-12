"""
    Generates image pairs of identities. One epoch is one run over all identities in the dataset (not all images!)
"""
import random
from collections import defaultdict
from functools import partial
import os

import PIL.Image
from torchvision.datasets import VisionDataset
from data.celeba_cropped import CelebaCropped

from shutil import copyfile

assert os.path.isdir("data")


class CelebaCroppedPairs(VisionDataset):
    cropped_base_folder = "celeba_cropped/img_align/"

    def __init__(self, split="train", transform=None, target_transform=None, download=False):
        super().__init__("data", transforms=None, transform=transform, target_transform=target_transform)

        if not os.path.isdir("data/celeba_cropped/"):
            # Initialize the dataset if it does not yet exist
            CelebaCropped(split, transform, target_transform, download)

        # Load filenames and splits
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[split]

        with open("data/celeba_cropped/list_eval_partition_morphing.txt", "r") as f:
            tuples = map(lambda s: tuple(s.split()), f.readlines())
            fnames = [fname for fname, fsplit in tuples if split is None or split == int(fsplit)]
            fnames = set(fnames)

        with open("data/celeba/identity_CelebA.txt", "r") as f:
            tuples = map(lambda s: tuple(s.split()), f.readlines())
            idents = defaultdict(lambda :[])
            for fname, ident in tuples:
                ident = int(ident)
                if fname in fnames:
                    idents[ident].append(fname)
        self.idents = idents
        self.ident_list = list(self.idents.keys())

    def generate_random_different_index(self, index):
        # Generate a random number between 0 and len(ident_list) - 2
        # (the minus 2 is because 1. randint includes the end value and 2. we do not want to include index.
        rand_index = random.randint(0, len(self.ident_list)-2)

        # Skip the current index, because we don't want it to be included
        if rand_index >= index:
            rand_index += 1

        return rand_index

    def __getitem__(self, index):
        index2 = self.generate_random_different_index(index)

        ident1 = self.ident_list[index]
        ident2 = self.ident_list[index2]

        fname1 = random.choice(self.idents[ident1])
        fname2 = random.choice(self.idents[ident2])

        return self.load_image(fname1), self.load_image(fname2)

    def load_image(self, name):
        X = PIL.Image.open(os.path.join(self.root, self.cropped_base_folder, name))

        if self.transform is not None:
            X = self.transform(X)

        return X

    def __len__(self):
        return len(self.ident_list)

if __name__ == "__main__":
    ds = CelebaCroppedPairs(download=True)

