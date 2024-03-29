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

from data.data_prep.celeba_sideways_detector import gen_aligned_faces

assert os.path.isdir("data")


class CelebaCropped(VisionDataset):
    cropped_base_folder = "celeba_cropped/img_align/"

    def __init__(self, split="train", transform=None, target_transform=None, download=False, morgan_like_filtering=False, validate_files=False, use_pair_split=True):
        super().__init__("data", transforms=None, transform=transform, target_transform=target_transform)

        # This option enables the train/valid/test splits used for the thesis
        self.use_pair_split = use_pair_split

        if not os.path.isdir("data/celeba"):
            # try to download celeba
            celeba = CelebA("data", split=split, transform=transform, target_transform=target_transform, download=download)


        # Check if files exist
        if not os.path.isdir("data/" + self.cropped_base_folder) or validate_files:
            if not download:
                raise IOError("Download is False, but the data does not exist")

            self.crop()

        if not os.path.isfile("data/celeba_cropped/list_eval_partition.txt"):
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

        if not use_pair_split and morgan_like_filtering and not os.path.isfile("data/celeba_cropped/list_eval_partition_filtered.txt"):
            # Get all aligned faces
            aligned = gen_aligned_faces()

            with open("data/celeba/list_eval_partition.txt", "r") as f:
                lines = f.readlines()
                splitted = [line.split(" ") for line in lines]
                outlines = []
                for fname, n in splitted:
                    if not os.path.isfile("data/" + self.cropped_base_folder + fname) or fname not in aligned:
                        continue
                    outlines.append("%s %s" % (fname, n))

            with open("data/celeba_cropped/list_eval_partition_filtered.txt", "w") as f:
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
        if use_pair_split:
            partition_file = "celeba_cropped/list_eval_partition_morphing.txt"
        else:
            partition_file = "celeba_cropped/list_eval_partition.txt" if not morgan_like_filtering else \
                "celeba_cropped/list_eval_partition_filtered.txt"
        splits = pandas.read_csv(fn(partition_file), delim_whitespace=True, header=None,
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
        if not os.path.exists("data/" + self.cropped_base_folder):
            os.mkdir("data/" + self.cropped_base_folder)

        # Crop images
        face_cropper.crop_images("data/celeba/img_align_celeba/", "data/" + self.cropped_base_folder + "/")


if __name__ == "__main__":
    ds = CelebaCropped(download=True, validate_files=True)

