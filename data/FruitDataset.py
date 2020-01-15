import os

import PIL
from torchvision.datasets import VisionDataset


class FruitDataset(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, only_original=False):
        super().__init__(root)
        self.fnames = os.listdir(root)
        if only_original:
            self.fnames = list([fname for fname in self.fnames if "original" in fname])
        self.only_original = only_original
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        index = index%len(self.fnames)
        img = self.load_image(self.fnames[index])
        return img, img

    def load_image(self, name):
        X = PIL.Image.open(os.path.join(self.root, name))

        if self.transform is not None:
            X = self.transform(X)

        return X

    def __len__(self):
        return len(self.fnames)*(10 if not self.only_original else 100)