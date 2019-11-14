"""
    Tests whether identities are split train/valid/test by default
"""
import os
from collections import defaultdict

assert os.path.isdir("data")

with open("data/celeba/identity_CelebA.txt", "r") as f:
    tuples = map(lambda s: tuple(s.split()), f.readlines())
    idents = {t[0]: int(t[1]) for t in tuples}

with open("data/celeba_cropped/list_eval_partition.txt", "r") as f:
    tuples = map(lambda s: tuple(s.split()), f.readlines())

    splits = defaultdict(lambda: set())
    for fname, part in tuples:
        splits[part].add(fname)

identity_splits = dict()
for part, fnames in splits.items():
    identity_splits[part] = {idents[fname] for fname in fnames}
    print(part, len(identity_splits[part]))

splitted_idents = None
for ident_set in identity_splits.values():
    if splitted_idents is None:
        splitted_idents = ident_set
    else:
        splitted_idents ^= ident_set

print(len(ident_set))