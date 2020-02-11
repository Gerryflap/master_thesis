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


class FRGCCropped(VisionDataset):
    cropped_base_folder = "frgc/cropped/"

    def __init__(self, transform=None, target_transform=None, download=False):
        super().__init__("data", transforms=None, transform=transform, target_transform=target_transform)

        fn = partial(os.path.join, self.root)

        partition_file = "frgc/list_eval_partition.txt"
        splits = pandas.read_csv(fn(partition_file), delim_whitespace=True, header=None,
                                 index_col=0)

        self.filename = splits.index.values

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.cropped_base_folder, self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)

        return X, []

    def __len__(self):
        return len(self.filename)


