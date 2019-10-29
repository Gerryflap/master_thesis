# This loader contains code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/celeba.py

from functools import partial

import PIL
import pandas
import torch
from torchvision.datasets import VisionDataset, CelebA
import os

from torchvision.datasets.utils import verify_str_arg

import data.data_prep.face_cropper as face_cropper
from shutil import copyfile

assert os.path.isdir("data")


class CelebaCropped(VisionDataset):
    cropped_base_folder = "celeba_cropped/img_align/"

    def __init__(self, split="train", transform=None, target_transform=None, download=False):
        super().__init__("data", transforms=None, transform=transform, target_transform=target_transform)

        if not os.path.isdir("data/celeba"):
            # try to download celeba
            celeba = CelebA("data", split=split, transform=transform, target_transform=target_transform, download=download)


        # Check if files exist
        if not os.path.isdir("data/" + self.cropped_base_folder):
            if not download:
                raise IOError("Download is False, but the data does not exist")

            self.crop()

        if not os.path.isfile("data/" + self.cropped_base_folder + "list_eval_partition.txt"):
            with open("data/celeba/list_eval_partition.txt", "r") as f:
                lines = f.readlines()
                splitted = [line.split(" ") for line in lines]
                outlines = []
                for fname, n in splitted:
                    if not os.path.isfile("data/" + self.cropped_base_folder + fname):
                        continue
                    outlines.append("%s %s" % (fname, n))

            with open("data/celeba_cropped/list_eval_partition.txt", "w") as f:
                f.writelines(outlines)

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pandas.read_csv(fn("celeba_cropped/list_eval_partition.txt"), delim_whitespace=True, header=None,
                                 index_col=0)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.cropped_base_folder, self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)

        return X, []

    def __len__(self):
        return len(self.filename)

    def crop(self):

        # Create the data directory
        os.mkdir("data/" + self.cropped_base_folder)

        # Crop images
        face_cropper.crop_images("data/celeba/img_align_celeba/", "data/" + self.cropped_base_folder + "/")


if __name__ == "__main__":
    ds = CelebaCropped(download=True)

