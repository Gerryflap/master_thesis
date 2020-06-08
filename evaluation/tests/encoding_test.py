"""
    This script tries to uncover why the results vary.
    Using n_jitters=10 will cause varying results for similar pictures, while n_jitters=0 or 1 does not.
"""
import random

import numpy as np
from collections import defaultdict

import face_recognition
from torchvision.transforms import transforms

from data.celeba_cropped_pairs_look_alike import CelebaCroppedPairsLookAlike


def to_numpy_img(img):
    img = np.moveaxis(img.detach().numpy(), 0, 2)
    img *= 255
    img = img.astype(np.uint8)
    return img


dataset = CelebaCroppedPairsLookAlike(split="valid", transform=transforms.ToTensor())

n_resamples = 3
n_images = len(dataset)

scores = defaultdict(lambda: [])


def compute_dist(img_tup):
    ((x1, x1_ref), (x2, x2_ref)) = img_tup
    x1_img = to_numpy_img(x1)
    x2_img = to_numpy_img(x2)

    x1_pos = face_recognition.face_locations(x1_img, number_of_times_to_upsample=2)
    x2_pos = face_recognition.face_locations(x2_img, number_of_times_to_upsample=2)

    x1_enc = face_recognition.face_encodings(x1_img, x1_pos, num_jitters=1)[0]
    x2_enc = face_recognition.face_encodings(x2_img, x2_pos, num_jitters=1)[0]
    dist = np.sqrt(np.sum((x1_enc - x2_enc) ** 2))
    return dist

for _ in range(n_resamples):
    indices = list(range(n_images))
    random.shuffle(indices)
    for i in indices:
        tup, _ = dataset[i]
        dist = compute_dist(tup)
        scores[i].append(dist)

for i, score in scores.items():
    val = score[0]
    for val2 in score[1:]:
        if val2 != val:
            print("Dissimilar scores in pair %d, %f vs %f"%(i, val, val2))
print("Done....")
print("If there is not output above the 'Done' message, all runs are consistent")
print(scores)
