import os
from collections import defaultdict

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import random

import torchvision

import face_recognition

# Make sure that the data directory exists and is a child of the current directory.
# If this isn't the case, make sure you're running from the root project folder
assert os.path.exists("data")

if not os.path.exists("data/celeba_cropped/embeddings.npy"):
    # Generate embeddings for all aligned faces
    import data.split_generation.embedding_generator as eg
    eg.run()

embeddings = np.load("data/celeba_cropped/embeddings.npy")

with open("data/celeba_cropped/embedding_file_order.txt") as f:
    fnames = f.readlines()
fnames = list(fname[:-1] for fname in fnames)

fname_to_i = {fname: i for i, fname in enumerate(fnames)}
fname_set = set(fnames)
print(fname_set)

# Make a dict mapping identity to a list of fnames
ident_to_fnames = defaultdict(lambda: [])
with open("data/celeba/identity_CelebA.txt") as f:
    lines = f.readlines()
    tuples = [tuple(l.split()) for l in lines]
    for fname, ident in tuples:
        # print(fname, ident)
        ident = int(ident)
        if fname in fname_set:
            ident_to_fnames[ident].append(fname)
print("Total idents: ", len(ident_to_fnames))

idents_in_range = list([ident for ident, fname_list in ident_to_fnames.items() if 4 < len(fname_list) < 10])
print("In range: ", len(idents_in_range))
for _ in range(5):
    ident = random.choice(idents_in_range)
    print("Picked identity %d"%ident)
    print("This identity has %d pictures"%len(ident_to_fnames[ident]))

    imgs = []
    embedding_list =[]
    indices = set()

    for fname in ident_to_fnames[ident]:
        img = Image.open("data/celeba_cropped/img_align/%s"%fname)
        img = to_tensor(img)
        imgs.append(img)
        index = fname_to_i[fname]
        indices.add(index)
        embedding = embeddings[index]
        if not np.isnan(embedding).any():
            embedding_list.append(embedding)

    grid = make_grid(torch.stack(imgs, dim=0), nrow=8)
    grid = np.moveaxis(grid.numpy(), 0, -1)
    plt.imshow(grid)
    plt.show()
    embedding_list = np.stack(embedding_list, axis=0)
    mean_embedding = embedding_list.mean(axis=0)

    distances = np.square(embeddings - mean_embedding).sum(axis=1)
    closest_indices = sorted(enumerate(distances), key=lambda e: e[1])

    closest_faces = []
    for index, dist in closest_indices:
        if len(closest_faces) > 15:
            break
        if index in indices:
            continue
        fname = fnames[index]
        img = Image.open("data/celeba_cropped/img_align/%s"%fname)
        img = to_tensor(img)
        closest_faces.append(img)

    grid = make_grid(torch.stack(closest_faces, dim=0), nrow=8)
    grid = np.moveaxis(grid.numpy(), 0, -1)
    plt.imshow(grid)
    plt.show()

    print(mean_embedding, embedding_list.var(axis=0))
    print()