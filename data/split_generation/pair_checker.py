"""
    This file exists to verify that the input image and reference images are indeed the same person to the FRS.
"""
import face_recognition
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

from data.celeba_cropped_pairs_look_alike import CelebaCroppedPairsLookAlike


def to_numpy_img(img):
    img = np.moveaxis(img.detach().numpy(), 0, 2)
    img *= 255
    img = img.astype(np.uint8)
    return img


dataset = CelebaCroppedPairsLookAlike(split="valid", transform=transforms.ToTensor())
loader = DataLoader(dataset, 64, shuffle=False)

inputs = []
references = []

for ((x1, x2), (x1ref, x2ref)), _ in loader:
    for img, ref_img in zip(torch.unbind(x1, dim=0), torch.unbind(x1ref, dim=0)):
        inputs.append(to_numpy_img(img))
        references.append(to_numpy_img(ref_img))

    for img, ref_img in zip(torch.unbind(x2, dim=0), torch.unbind(x2ref, dim=0)):
        inputs.append(to_numpy_img(img))
        references.append(to_numpy_img(ref_img))

face_locations_inputs = face_recognition.batch_face_locations(inputs)
face_locations_references = face_recognition.batch_face_locations(references)


def get_encodings(faces_list, face_locations):
    face_encodings = []
    for face, face_location in zip(faces_list, face_locations):
        if len(face_location) != 1:
            nans = np.zeros((128,), dtype=np.float32)
            nans.fill(np.nan)
            face_encodings.append(nans)
        else:
            face_enc = face_recognition.face_encodings(face, face_location)[0]
            face_encodings.append(face_enc)
    return face_encodings


input_enc = np.stack(get_encodings(inputs, face_locations_inputs), axis=0)
ref_enc = np.stack(get_encodings(references, face_locations_references), axis=0)

distances = np.sqrt(np.sum(np.square(input_enc - ref_enc), axis=1))
print("Encountered %d nan distances"%np.sum(np.isnan(distances)))
distances = distances[~np.isnan(distances)]


plt.hist(distances, bins=30)
plt.show()

print("Mean distance: ", distances.mean())
print("Max distance: ", distances.max())
print("Min distance: ", distances.min())
print("Under threshold: ", np.sum(distances < 0.6)/distances.shape[0])