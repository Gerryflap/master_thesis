"""
    This program loads a model and can be used to compute the MMPMR over the validation or test set
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid, save_image

from data.celeba_cropped_pairs_look_alike import CelebaCroppedPairsLookAlike
from evaluation.metrics.evaluation_metrics import mmpmr, relative_morph_distance
from models.morphing_encoder import MorphingEncoder
import face_recognition
from util.output import init_experiment_output_dir

def to_numpy_img(img, tanh_mode):
    if tanh_mode:
        img = (img + 1)/2
    img = np.moveaxis(img.detach().numpy(), 0, 2)
    img *= 255
    img = img.astype(np.uint8)
    return img

parser = argparse.ArgumentParser(description="Morph inspection tool.")
parser.add_argument("--batch_size", action="store", type=int, default=24,
                    help="Changes the batch size used when evaluating. "
                         "Higher might be faster, but will also take more memory")
parser.add_argument("--experiment_path", action="store", type=str,
                    help="Path to the experiment. Either this or --parameter_path is required."
                         "Experiment path will simply take parameters out of the params/all_epochs folder.")
parser.add_argument("--parameter_path", action="store", type=str,
                    help="Path to the parameter folder (usually called either all_epochs or epoch_<number>). "
                         "Either this or --experiment_path is required."
                         "With this option you can select the epoch yourself instead of auto-picking all_epochs")
parser.add_argument("--max_output_batches", action="store", type=int, default=None,
                    help="If defined, limits the amount of rows in the output image")
parser.add_argument("--res", action="store", type=int, default=64,
                    help="Image resolution. 64 (for 64x64) by default. 28 if you're loading a 28x28 model")
parser.add_argument("--tanh", action="store_true",  default=False,
                    help="Has to be used if the model is a tanh model instead of sigmoid")
parser.add_argument("--test", action="store_true", default=False, help="Switches to the test set")
parser.add_argument("--decoder_filename", action="store", type=str, default="Gx.pt",
                    help="Filename of the decoder/generator/Gx network. "
                         "Usually this option can be left at the default value, which is Gx.pt")
parser.add_argument("--encoder_filename", action="store", type=str, default="Gz.pt",
                    help="Filename of the encoder/Gz network. "
                         "Usually this option can be left at the default value, which is Gz.pt")
parser.add_argument("--use_z_mean", action="store_true", default=False,
                    help="Uses z = z_mean instead of sampling from q(z|x)"
                    )
parser.add_argument("--cuda", action="store_true", default=False, help="When this flag is present, cuda is used")
parser.add_argument("--eval", action="store_true", default=False,
                    help="When this flag is present, the models are put in evaluation mode. This affects BatchNorm")
parser.add_argument("--train", action="store_true", default=False,
                    help="When this flag is present, the models are put in train mode. This affects BatchNorm")
parser.add_argument("--visualize", action="store_true", default=False,
                    help="When this flag is present, a matplotlib visualization is shown with the best and worst morphs")
args = parser.parse_args()

if args.test:
    print("WARNING! Test set is enabled. This is only allowed when evaluating the model!")
    response = input("Please type \"use test\" in order to continue: \n")
    if response != "use test":
        print("Input did not match the required string: exiting...")
        exit()

if args.parameter_path is None and args.experiment_path is None:
    raise ValueError("Not path specified. Please specify either parameter_path or experiment_path")

if args.experiment_path is not None:
    param_path = os.path.join(args.experiment_path, "params", "all_epochs")
else:
    param_path = args.parameter_path

device = torch.device("cpu") if not args.cuda else torch.device("cuda")
Gx = torch.load(os.path.join(param_path, args.decoder_filename), map_location=device)
Gz = torch.load(os.path.join(param_path, args.encoder_filename), map_location=device)
if not isinstance(Gz, MorphingEncoder):
    print("Gz is not a subclass of MorphingEncoder! Morphing is now done the MorGAN way.")
    manual_morph = True
else:
    manual_morph = False
if args.eval:
    Gx.eval()
    Gz.eval()

if args.train:
    Gx.train()
    Gz.train()

trans = []
if args.res != 64:
    trans.append(transforms.Resize(args.res))

trans.append(transforms.ToTensor())

if args.tanh:
    trans.append(transforms.Lambda(lambda img: img*2.0 - 1.0))

dataset = CelebaCroppedPairsLookAlike(split="test" if args.test else "valid", transform=transforms.Compose(trans))
loader = DataLoader(dataset, args.batch_size, shuffle=False)

x1_list = []
morph_list = []
x2_list = []

for i, batch in enumerate(loader):
    if args.max_output_batches is not None and i >= args.max_output_batches:
        break
    ((x1, x2), (x1_comp, x2_comp)), idents = batch

    for img in torch.unbind(x1_comp, dim=0):
        x1_list.append(to_numpy_img(img, args.tanh))

    for img in torch.unbind(x2_comp, dim=0):
        x2_list.append(to_numpy_img(img, args.tanh))

    if args.cuda:
        x1 = x1.cuda()
        x2 = x2.cuda()


    if manual_morph:
        z1, z1m, _ = Gz(x1)
        z2, z2m, _ = Gz(x2)

        if args.use_z_mean:
            z1, z2 = z1m, z2m

        z_morph = 0.5*(z1 + z2)
    else:
        z_morph = Gz.morph(x1, x2, use_mean=args.use_z_mean)
    x_morph = Gx(z_morph)

    if args.cuda:
        x_morph = x_morph.cpu()

    for img in torch.unbind(x_morph, dim=0):
        morph_list.append(to_numpy_img(img, args.tanh))

n_morphs = len(morph_list)
faces_list = x1_list + x2_list + morph_list
face_locations = face_recognition.batch_face_locations(faces_list)
face_encodings = []
for face, face_location in zip(faces_list, face_locations):
    if len(face_location) != 1:
        nans = np.zeros((128,), dtype=np.float32)
        nans.fill(np.nan)
        face_encodings.append(nans)
    else:
        face_enc = face_recognition.face_encodings(face, face_location)[0]
        face_encodings.append(face_enc)

x1_list = np.stack(x1_list, axis=0)
x2_list = np.stack(x2_list, axis=0)
morph_list = np.stack(morph_list, axis=0)

x1_enc = np.stack(face_encodings[:n_morphs], axis=0)
x2_enc = np.stack(face_encodings[n_morphs:2*n_morphs], axis=0)
morphs_enc = np.stack(face_encodings[2*n_morphs:], axis=0)

# Filter any rows with nan embeddings in the x1 and x2
# TODO: Find a better solution to this!
not_nan_indices = ~(np.isnan(np.sum(x1_enc, axis=1)) + np.isnan(np.sum(x2_enc, axis=1)))

print("WARNING! Due to undetectable faces in the dataset, %d images have been dropped!"%int(np.sum(~not_nan_indices)))

x1_list = x1_list[not_nan_indices]
x2_list = x2_list[not_nan_indices]
morph_list = morph_list[not_nan_indices]

x1_enc = x1_enc[not_nan_indices]
x2_enc = x2_enc[not_nan_indices]
morphs_enc = morphs_enc[not_nan_indices]


# Assert that there are no nans in the comparison faces
# assert not (np.isnan(x1_enc).any() or np.isnan(x2_enc).any())


# Compute euclidean distances between x1 and the morph and x2 and the morph
dist_x1 = np.sqrt(np.sum(np.square(x1_enc - morphs_enc), axis=1))
dist_x2 = np.sqrt(np.sum(np.square(x2_enc - morphs_enc), axis=1))
dist_x1_x2 = np.sqrt(np.sum(np.square(x1_enc - x2_enc), axis=1))


s = np.stack([dist_x1, dist_x2], axis=1)

mmpmr_value = mmpmr(s, threshold=0.6)
rmd, rmd_values = relative_morph_distance(dist_x1, dist_x2, dist_x1_x2)

print("===== RESULTS =====")
print()
print("Computed MMPMR: ", mmpmr_value)
print("Computed Mean RMD: ", rmd)
print()
print("===================")

if args.visualize:
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.titlesize': 8})

    f = plt.figure(figsize=(20, 100), dpi=120)
    axs = f.subplots(5, 2)

    max_distances = np.max(s, axis=1)
    sorted_indices = np.argsort(max_distances)
    top5 = sorted_indices[:5]
    worst5 = sorted_indices[-5:]

    for plot_index, i in enumerate(top5):
        img = np.concatenate([x1_list[i], morph_list[i], x2_list[i]], axis=1)
        axs[plot_index, 0].imshow(img)
        axs[plot_index, 0].set_title("No %d of top 5, Max distance: %.2f" % (plot_index+1, max_distances[i]))
        axs[plot_index, 0].axis('off')

    for plot_index, i in enumerate(worst5):
        img = np.concatenate([x1_list[i], morph_list[i], x2_list[i]], axis=1)
        axs[plot_index, 1].imshow(img)
        axs[plot_index, 1].set_title("No %d of bottom 5, Max distance: %.2f" % (plot_index+1, max_distances[i]))

        axs[plot_index, 1].axis('off')

    plt.subplots_adjust(hspace=1.0)
    plt.show()

    plt.hist(max_distances, bins=60, range=(0.0, 1.2))
    plt.title("Max euclidean distances")
    plt.show()

    plt.hist(rmd_values, bins=60)
    plt.title("RMD Values")
    plt.show()