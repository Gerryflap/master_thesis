"""
    Generates image triplets with (img_of_person1, other_img_of_person1, img_of_person2).
    This dataset counts an epoch as every image having been the first image in the tuple once.
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


class CelebaCroppedTriplets(VisionDataset):
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
            self.fnames = list(fnames)
            fnames = set(self.fnames)

        with open("data/celeba/identity_CelebA.txt", "r") as f:
            tuples = map(lambda s: tuple(s.split()), f.readlines())
            idents = defaultdict(lambda :[])
            self.fname_to_ident = dict()
            self.fname_to_index_in_ident_image_list = dict()
            for fname, ident in tuples:
                ident = int(ident)
                if fname in fnames:
                    self.fname_to_index_in_ident_image_list[fname] = len(idents[ident])
                    idents[ident].append(fname)
                    self.fname_to_ident[fname] = ident
        self.idents = {k: v for k, v in idents.items() if len(v) > 1}
        self.fnames = []

        for fname_list in self.idents.values():
            self.fnames += fname_list

        self.ident_list = list(self.idents.keys())
        self.ident_indices = {ident: i for i, ident in enumerate(self.ident_list)}

    def generate_random_different_index(self, index, list_length=None):
        if list_length is None:
            list_length = len(self.ident_list)

        if list_length == 1:
            raise ValueError("Cannot choose a different element from 0 options!")

        if list_length == 2:
            return (index + 1) % 2

        # Generate a random number between 0 and len(ident_list) - 2
        # (the minus 2 is because 1. randint includes the end value and 2. we do not want to include index.
        rand_index = random.randint(0, list_length-2)

        # Skip the current index, because we don't want it to be included
        if rand_index >= index:
            rand_index += 1

        return rand_index

    def __getitem__(self, index):
        fname = self.fnames[index]
        ident1 = self.fname_to_ident[fname]

        ident_index3 = self.generate_random_different_index(self.ident_indices[ident1])

        ident3 = self.ident_list[ident_index3]

        fname1 = fname
        if len(self.idents[ident1]) == 1:
            print(self.idents[ident1])
        image_index2 = self.generate_random_different_index(
            self.fname_to_index_in_ident_image_list[fname],
            len(self.idents[ident1]))
        fname2 = self.idents[ident1][image_index2]
        fname3 = random.choice(self.idents[ident3])

        return self.load_image(fname1), self.load_image(fname2), self.load_image(fname3)

    def load_image(self, name):
        X = PIL.Image.open(os.path.join(self.root, self.cropped_base_folder, name))

        if self.transform is not None:
            X = self.transform(X)

        return X

    def __len__(self):
        return len(self.fnames)


