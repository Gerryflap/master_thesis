"""
    FRGC is expected to be in data/frgc/img
    There it should contain a filtered set of images with image and embeddings as a npy file.

    This script generates:
    - Cropped 64x64 images
    - A list of splits (The dataset will not be split, so everything is "split" 0)
    - A file containing all identities and filenames
    - A file containing the filenames in order of embeddings
    - A npy file containing all embeddings

    All this stuff will go to data/frgc except for the cropped images, which will go to data/frgc/cropped
"""
import os
from data.data_prep import face_cropper
import numpy as np

# Check if we're operating from the project root
assert os.path.exists("data")

# Cropping
if not os.path.exists("data/frgc/cropped"):
    os.mkdir("data/frgc/cropped")

print("Generating cropped images...")
face_cropper.crop_images("data/frgc/img/", "data/frgc/cropped/")
print("Done generating cropped images...")

print("Generating fname files and merging embeddings into one file...")

# Loading fnames and creating fname -> ident mapping
fnames = os.listdir("data/frgc/cropped")
fnames_to_ident = {fname: fname.split("d")[0] for fname in fnames}

# Generating embedding file and a list of image fnames in order of the embedding file
filenames_without_ext = [fname[:-4] for fname in os.listdir("data/frgc/img/") if fname[-3:] == "npy"]
tuples = [(fname + ".jpg", fname + ".npy") for fname in filenames_without_ext]

embedding_fname_order = []
embeddings = []
for img_fname, embedding_fname in tuples:
    embedding_fname_order.append(img_fname)
    embedding = np.load(os.path.join("data/frgc/img/", embedding_fname))
    embeddings.append(embedding)

embeddings = np.stack(embeddings, axis=0)
print("Done, saving files...")
# Saving all files
path = "data/frgc/"

# All filenames
with open(os.path.join(path, "list_eval_partition.txt"), "w") as f:
    f.write("\n".join([fname + " 0" for fname in fnames])+ "\n")

with open(os.path.join(path, "identity_frgc.txt"), "w") as f:
    f.write("\n".join([fname + " " + identity for fname, identity in fnames_to_ident.items()])+ "\n")

with open(os.path.join(path, "embedding_file_order.txt"), "w") as f:
    f.write("\n".join(embedding_fname_order) + "\n")

np.save(os.path.join(path, "embeddings.npy"), embeddings)
print("Finished...")