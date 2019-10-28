from torchvision.datasets import CelebA
import os
import data.data_prep.face_cropper as face_cropper

assert os.path.isdir("data")


class CroppedImageDataset(CelebA):
    cropped_base_folder = "celeba_cropped"

    def __init__(self, split="train", target_type="attr", transform=None, target_transform=None, download=False):
        super().__init__("data", split, target_type, transform, target_transform, download)

        # Check if files exist
        if not os.path.isdir("data/"+self.cropped_base_folder):
            if not download:
                raise IOError("Download is False, but the data does not exist")

            self.crop()

    def __getitem__(self, index):
        super().__getitem__(index)

    def __len__(self):
        super().__len__()

    def crop(self):

        # Create the data directory
        os.mkdir("data/"+self.cropped_base_folder)

        # Crop images
        face_cropper.crop_images("data/celeba/img_align_celeba/", "data/"+self.cropped_base_folder+"/")


if __name__ == "__main__":
    ds = CroppedImageDataset(download=True)

