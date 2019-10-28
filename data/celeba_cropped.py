from torchvision.datasets import CelebA
import os

assert os.path.isdir("data")


class CroppedImageDataset(CelebA):
    cropped_base_folder = "celeba_cropped"

    def __init__(self, split="train", target_type="attr", transform=None, target_transform=None, download=False):
        super().__init__("data", split, target_type, transform, target_transform, download)

        # Check if files exist
        if not os.path.isdir("data/celeba_cropped"):
            if not download:
                raise IOError("Download is False, but the data does not exist")

    def __getitem__(self, index):
        super().__getitem__(index)

    def __len__(self):
        super().__len__()


if __name__ == "__main__":
    ds = CroppedImageDataset(download=True)
