"""
    This file is responsible for generating the official splits used for all experiments.
    Since it might not be deterministic, the official splits are already in the repositories
"""

import os
from collections import defaultdict

import numpy as np
import json

# Configures the amount of PAIRS of identities to make.
n_validation = 250
n_test = 500

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

idents_in_range = list([ident for ident, fname_list in ident_to_fnames.items() if 3 <= len(fname_list) <= 6])
print("In range: ", len(idents_in_range))

ident_to_i = {ident: i for i, ident in enumerate(idents_in_range)}

if len(idents_in_range) < n_test * 2 + n_validation * 2:
    raise ValueError("The specified dataset sizes are too high, there are not enough identities to use!")


def compute_mean_identity_embedding(ident):
    embedding_list = []
    for fname in ident_to_fnames[ident]:
        index = fname_to_i[fname]
        embedding = embeddings[index]
        if not np.isnan(embedding).any():
            embedding_list.append(embedding)

    embedding_list = np.stack(embedding_list, axis=0)
    mean_embedding = embedding_list.mean(axis=0)
    return mean_embedding


# Make an array of the mean embeddings of an identity, in the order of idents_in_range
mean_embeddings = np.stack(map(compute_mean_identity_embedding, idents_in_range), axis=0)

# build a set that contains all remaining identity indices
# (these are indices for the idents_in_range list, not the identity identifiers themselves)
ident_indices = set(range(len(idents_in_range)))


def get_top_2_images(ident):
    # Finds the 2 images that are closest to the mean in embedding space
    # It also tries to end up with images with different embeddings to try and avoid duplicates
    mean_embedding = mean_embeddings[ident_to_i[ident]]
    image_fnames = ident_to_fnames[ident]
    fname_indices = [fname_to_i[fn] for fn in image_fnames]
    image_embeddings = np.stack([embeddings[fname_index] for fname_index in fname_indices], axis=0)
    distances = np.square(mean_embedding - image_embeddings).sum(axis=1)
    sort = np.argsort(distances)
    best_image = sort[0]
    second_best_image = sort[1]

    for j in range(sort.shape[0]):
        if (image_embeddings[best_image] != image_embeddings[sort[j]]).any():
            second_best_image = sort[j]

    return image_fnames[best_image], image_fnames[second_best_image]


test_set_identities = set()
valid_set_identities = set()
test = []
valid = []

for i in range(n_test + n_validation):
    ident_index = ident_indices.pop()
    distances = np.square(mean_embeddings - mean_embeddings[ident_index]).sum(axis=1)
    sort = np.argsort(distances)

    best_paired_ident_index = None
    for j in range(sort.shape[0]):
        if sort[j] != ident_index and sort[j] in ident_indices:
            best_paired_ident_index = sort[j]
            break
    ident_indices.remove(best_paired_ident_index)

    ident_first = idents_in_range[ident_index]
    top2_first = get_top_2_images(ident_first)
    first = {"ident": ident_first, "morph_in": top2_first[0], "comparison_pic": top2_first[1]}

    ident_second = idents_in_range[best_paired_ident_index]
    top2_second = get_top_2_images(ident_second)
    second = {"ident": ident_second, "morph_in": top2_second[0], "comparison_pic": top2_second[1]}

    if i < n_test:
        test_set_identities.add(ident_first)
        test_set_identities.add(ident_second)
        test.append((first, second))
    else:
        valid_set_identities.add(ident_first)
        valid_set_identities.add(ident_second)
        valid.append((first, second))

total_idents = set(ident_to_fnames.keys())
train_set_identities = total_idents - test_set_identities - valid_set_identities
print("Total identities: ", len(total_idents))
print("Train identities: ", len(train_set_identities))
print("Valid identities: ", len(valid_set_identities))
print("Test identities: ", len(test_set_identities))



eval_partition = ""
for idents, value in [(train_set_identities, 0), (valid_set_identities, 1), (test_set_identities, 2)]:
    for ident in idents:
        for fname in ident_to_fnames[ident]:
            eval_partition += "%s %d\n" % (fname, value)

with open("data/celeba_cropped/list_eval_partition_morphing.txt", "w") as f:
    f.write(eval_partition)


def save_pairs(path_to_file, pairs):
    with open(path_to_file, "w") as f:
        json.dump(pairs, f)


save_pairs("data/celeba_cropped/valid_pairs.json", valid)
save_pairs("data/celeba_cropped/test_pairs.json", test)
