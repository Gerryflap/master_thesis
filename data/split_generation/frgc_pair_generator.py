"""
    This File uses the existing splits for train/valid/test and creates new pairs for the specified set
    that are guaranteed to have at most 0.4 distance between the images of the same identity.
    Any images that do not fit this requirement are thrown out.
"""
import json
from collections import defaultdict

import numpy as np

# 0 = train, 1 = valid, 2 = test
chosen_split = "0"
threshold_between_morph_pairs = 1.0

embeddings = np.load("data/frgc/embeddings.npy")
with open("data/frgc/embedding_file_order.txt") as f:
    fnames = f.readlines()
fnames = list(fname[:-1] for fname in fnames)
fname_to_index = {fname: i for i, fname in enumerate(fnames)}

with open("data/frgc/list_eval_partition.txt") as f:
    fnames_with_split = f.readlines()
    fnames_split = [l.split() for l in fnames_with_split]
    fnames_in_split = [fname for fname, split in fnames_split if split == chosen_split]
fname_set = set(fnames_in_split)

ident_to_fnames = defaultdict(lambda: [])
with open("data/frgc/identity_frgc.txt") as f:
    lines = f.readlines()
    tuples = [tuple(l.split()) for l in lines]
    for fname, ident in tuples:
        # print(fname, ident)
        if fname in fname_set:
            ident_to_fnames[ident].append(fname)

ident_list = list(ident_to_fnames.keys())
ident_to_ident_list_index = {ident: i for i, ident in enumerate(ident_list)}
print("Total idents: ", len(ident_to_fnames))

def euclidean_distance(emb1, emb2):
    return np.sqrt(np.sum(np.square(emb1 - emb2), axis=-1))


def find_top2_image_pair(ident):
    print(ident_to_fnames[ident])
    print([fname_to_index[fname] if fname in fname_to_index.keys() else None  for fname in ident_to_fnames[ident]])

    ident_embeddings = np.stack([embeddings[fname_to_index[fname]] for fname in ident_to_fnames[ident]], axis=0)

    indices = None
    for first_index in range(ident_embeddings.shape[0]):
        for second_index in range(first_index, ident_embeddings.shape[0]):
            if first_index == second_index:
                continue
            dist = euclidean_distance(ident_embeddings[first_index], ident_embeddings[second_index])
            if dist < 0.4 and dist != 0:
                indices = (ident_to_fnames[ident][first_index], ident_to_fnames[ident][second_index])
                indices = (fname_to_index[indices[0]], fname_to_index[indices[1]])
    return indices

def get_ident_dict(ident):
    fname_indices = image_pairs[ident]
    morph_in = fnames[fname_indices[0]]
    comparison_pic = fnames[fname_indices[1]]
    return {"ident": ident, "morph_in": morph_in, "comparison_pic": comparison_pic}


image_pairs = {ident: find_top2_image_pair(ident) for ident in ident_list}
input_image_embeddings = np.stack([embeddings[image_pairs[ident][0]] if image_pairs[ident] is not None else np.full((128,), np.nan) for ident in ident_list], axis=0)
ref_image_embeddings = np.stack([embeddings[image_pairs[ident][1]] if image_pairs[ident] is not None else np.full((128,), np.nan) for ident in ident_list], axis=0)

pairs = []
unused_idents = set(ident_list)
while len(unused_idents) >= 2:
    ident = unused_idents.pop()
    ident_index = ident_to_ident_list_index[ident]
    embedding = input_image_embeddings[ident_index]
    ref_embedding = input_image_embeddings[ident_index]
    inp_distances = euclidean_distance(embedding, input_image_embeddings)
    ref_distances = euclidean_distance(embedding, ref_image_embeddings)

    # Switched from max to only input distances.
    # This is more fair since at the time of creating the morph you don't have the references yet.
    # distances = np.maximum(inp_distances, ref_distances)
    distances = inp_distances

    sort = np.argsort(distances)

    for other_index in sort:
        if other_index == ident_index:
            continue

        other_ident = ident_list[other_index]

        if other_ident not in unused_idents:
            continue

        if distances[other_index] <= threshold_between_morph_pairs:
            print("Pairing ", ident, other_ident)
            # We have found a suitable candidate
            unused_idents.remove(other_ident)

            pairs.append([get_ident_dict(ident), get_ident_dict(other_ident)])
            break
        else:
            # We've passed the threshold, no suitable candidate can be found for this identity
            break

print("Made %d pairs, saving..."%len(pairs))
with open("data/frgc/pairs.json", 'w') as f:
    json.dump(pairs, f)
