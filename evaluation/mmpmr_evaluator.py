"""
    This program loads a model and can be used to compute the MMPMR over the validation or test set

    The model used by face_recognition is from dlib.
    More info about the model at: http://dlib.net/face_recognition.py.html
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import json

from data.celeba_cropped_pairs_look_alike import CelebaCroppedPairsLookAlike
from data.frgc_cropped_pairs_look_alike import FRGCPairsLookAlike
from evaluation.metrics.evaluation_metrics import mmpmr, relative_morph_distance
from evaluation.util.gradient_descend_on_z import optimize_z_batch, optimize_z_batch_recons
from models.morphing_encoder import MorphingEncoder
import face_recognition
import seaborn as sns

from util.interpolation import torch_slerp
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
parser.add_argument("--discriminator_filename", action="store", type=str, default="D.pt",
                    help="Filename of the discriminator network. Only used with gradient descend morphing. "
                         "Default is D.pt")
parser.add_argument("--use_z_mean", action="store_true", default=False,
                    help="Uses z = z_mean instead of sampling from q(z|x)"
                    )
parser.add_argument("--cuda", action="store_true", default=False, help="When this flag is present, cuda is used")
parser.add_argument("--eval", action="store_true", default=False,
                    help="When this flag is present, the models are put in evaluation mode. This affects BatchNorm")
parser.add_argument("--train", action="store_true", default=False,
                    help="When this flag is present, the models are put in train mode. This affects BatchNorm")
parser.add_argument("--visualize", action="store_true", default=False,
                    help="When this flag is present, a matplotlib visualization is written to the output folder"
                         " with the best and worst morphs")
parser.add_argument("--frgc", action="store_true", default=False,
                    help="When this flag is present, the FRGC dataset will be used for evaluation")
parser.add_argument("--gradient_descend_dis_l", action="store_true", default=False,
                    help="When this flag is present, "
                         "z_morph will be optimized further using gradient descend with dis_l loss")
parser.add_argument("--gradient_descend_dis_l_recon", action="store_true", default=False,
                    help="When this flag is present, "
                         "z1 and z2 will be optimized further using gradient descend with dis_l loss")
parser.add_argument("--slerp", action="store_true", default=False,
                    help="Uses slerp interpolation in latent space. "
                         "This is supposed to work better for normal distributions")
parser.add_argument("--force_linear_morph", action="store_true", default=False,
                    help="Forces linear morphing (the way morgan does it). "
                         "Can be useful to use as a baseline on a model that implements a different morph function")
parser.add_argument("--shuffle", action="store_true", default=False,
                    help="Shuffles the dataset. THIS IS EXPERIMENTAL AND MIGHT NOT YIELD CORRECT RESULTS!")
parser.add_argument("--enable_batched_face_detection", action="store_true", default=False,
                    help="Enables face detection in batches. This might speed the script up very slightly.")
parser.add_argument("--fast_mode", action="store_true", default=False,
                    help="Does image encoding way faster, but results will be less accurate.")
args = parser.parse_args()

if args.test:
    print("WARNING! Test set is enabled. This is only allowed when evaluating the model!")
    # Test set lock has been removed as of the 29th of April 2020 to allow for running many tests sequentially without
    # intervention.
    #
    # response = input("Please type \"use test\" in order to continue: \n")
    # if response != "use test":
    #     print("Input did not match the required string: exiting...")
    #     exit()

if args.parameter_path is None and args.experiment_path is None:
    raise ValueError("No path specified. Please specify either parameter_path or experiment_path")

if args.experiment_path is not None:
    param_path = os.path.join(args.experiment_path, "params", "all_epochs")
else:
    param_path = args.parameter_path

device = torch.device("cpu") if not args.cuda else torch.device("cuda")
Gx = torch.load(os.path.join(param_path, args.decoder_filename), map_location=device)
Gz = torch.load(os.path.join(param_path, args.encoder_filename), map_location=device)

morph_net = None
if os.path.exists(os.path.join(param_path, "morph_net.pt")):
    if args.force_linear_morph or args.slerp:
        print("Morph network found but not used due to force_linear_morph or slerp.")
    else:
        print("Found morph network in folder. It will be used for morphing. To disable this use --force_linear_morph")
        morph_net = torch.load(os.path.join(param_path, "morph_net.pt"), map_location=device)

if args.gradient_descend_dis_l or args.gradient_descend_dis_l_recon:
    D = torch.load(os.path.join(param_path, args.discriminator_filename), map_location=device)

if not isinstance(Gz, MorphingEncoder):
    print("Gz is not a subclass of MorphingEncoder! Morphing is now done the MorGAN way.")
    manual_morph = True
elif args.force_linear_morph:
    manual_morph = True
else:
    manual_morph = False

output_path = init_experiment_output_dir("celeba64" if not args.frgc else "frgc64", "model_evaluation", args)


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
split = "valid"

if args.test:
    split = "test"

if args.frgc:
    dataset = FRGCPairsLookAlike(transform=transforms.Compose(trans))
else:
    dataset = CelebaCroppedPairsLookAlike(split=split, transform=transforms.Compose(trans))
loader = DataLoader(dataset, args.batch_size, shuffle=args.shuffle)

print("Generating Morphs...")
x1_list = []
x2_list = []
morph_list = []
x1_recon_list = []
x2_recon_list = []
x1_inp_list = []
x2_inp_list = []

for i, batch in enumerate(loader):
    if args.max_output_batches is not None and i >= args.max_output_batches:
        break
    ((x1, x2), (x1_comp, x2_comp)), idents = batch

    if args.cuda:
        x1 = x1.cuda()
        x2 = x2.cuda()

    if args.slerp:
        # Use slerp interpolation
        z1, z1m, _ = Gz(x1)
        z2, z2m, _ = Gz(x2)

        if args.use_z_mean:
            z1, z2 = z1m, z2m
        z_morph = torch_slerp(0.5, z1, z2, dim=1)
    elif morph_net is not None:
        z1, z1m, _ = Gz(x1)
        z2, z2m, _ = Gz(x2)

        if args.use_z_mean:
            z1, z2 = z1m, z2m
        z_morph = morph_net.morph_zs(z1, z2)
    elif manual_morph:
        # Use linear interpolation (Like MorGAN)
        z1, z1m, _ = Gz(x1)
        z2, z2m, _ = Gz(x2)

        if args.use_z_mean:
            z1, z2 = z1m, z2m

        z_morph = 0.5*(z1 + z2)
    else:
        # Use provided morph method (this will often also be linear)
        z_morph, z1, z2 = Gz.morph(x1, x2, use_mean=args.use_z_mean, return_all=True)

    if args.gradient_descend_dis_l:
        z_morph, _ = optimize_z_batch(Gx, x1, x2, starting_z=z_morph, dis_l_D=D, n_steps=500)
        print("Batch %d/%d done..." % (i+1, len(loader)))

    if args.gradient_descend_dis_l_recon:
        (z1, z2, z_morph), _ = optimize_z_batch_recons(Gx, x1, x2, starting_zs=(z1, z2), dis_l_D=D, n_steps=500)
        print("Batch %d/%d done..." % (i+1, len(loader)))

    x1_recon = Gx(z1)
    x2_recon = Gx(z2)
    x_morph = Gx(z_morph)

    if args.cuda:
        x1_recon = x1_recon.cpu()
        x2_recon = x2_recon.cpu()
        x_morph = x_morph.cpu()
        x1 = x1.cpu()
        x2 = x2.cpu()

    for img in torch.unbind(x1_comp, dim=0):
        x1_list.append(to_numpy_img(img, args.tanh))

    for img in torch.unbind(x2_comp, dim=0):
        x2_list.append(to_numpy_img(img, args.tanh))

    for img in torch.unbind(x_morph, dim=0):
        morph_list.append(to_numpy_img(img, args.tanh))

    for img in torch.unbind(x1_recon, dim=0):
        x1_recon_list.append(to_numpy_img(img, args.tanh))

    for img in torch.unbind(x2_recon, dim=0):
        x2_recon_list.append(to_numpy_img(img, args.tanh))

    for img in torch.unbind(x1, dim=0):
        x1_inp_list.append(to_numpy_img(img, args.tanh))

    for img in torch.unbind(x2, dim=0):
        x2_inp_list.append(to_numpy_img(img, args.tanh))
print("Done.")

n_morphs = len(morph_list)
faces_list = x1_list + x2_list + morph_list + x1_recon_list + x2_recon_list + x1_inp_list + x2_inp_list

print("Detecting faces in all input and morph images...")
if args.enable_batched_face_detection:
    face_locations = face_recognition.batch_face_locations(faces_list)
else:

    face_locations = [face_recognition.face_locations(face, number_of_times_to_upsample=2) for face in faces_list]
    print(face_locations[0])
print("Done.")


print("Computing embedding vectors for all input and morph images...")
# face_encodings = []
# for face, face_location in zip(faces_list, face_locations):


def compute_encoding(tup):
    face, face_location = tup
    if len(face_location) != 1:
        nans = np.zeros((128,), dtype=np.float32)
        nans.fill(np.nan)
        return nans
    else:
        face_enc = face_recognition.face_encodings(face, face_location, num_jitters=10 if not args.fast_mode else 1)[0]
        return face_enc


pool = Pool(processes=16)
face_encodings = pool.map(compute_encoding, zip(faces_list, face_locations))
pool.close()


print("Done.")
print("Collecting data and computing statistics...")
x1_list = np.stack(x1_list, axis=0)
x2_list = np.stack(x2_list, axis=0)
morph_list = np.stack(morph_list, axis=0)
x1_recon_list = np.stack(x1_recon_list, axis=0)
x2_recon_list = np.stack(x2_recon_list, axis=0)

x1_enc = np.stack(face_encodings[:n_morphs], axis=0)
x2_enc = np.stack(face_encodings[n_morphs:2*n_morphs], axis=0)
morphs_enc = np.stack(face_encodings[2*n_morphs:3*n_morphs], axis=0)
x1_recon_enc = np.stack(face_encodings[3*n_morphs:4*n_morphs], axis=0)
x2_recon_enc = np.stack(face_encodings[4*n_morphs:5*n_morphs], axis=0)
x1_inp_enc = np.stack(face_encodings[5*n_morphs:6*n_morphs], axis=0)
x2_inp_enc = np.stack(face_encodings[6*n_morphs:], axis=0)


# Filter any rows with nan embeddings in the x1 and x2
not_nan_indices = ~(np.isnan(np.sum(x1_enc, axis=1)) + np.isnan(np.sum(x2_enc, axis=1)))

print("WARNING! Due to undetectable faces in the dataset, %d images have been dropped!"%int(np.sum(~not_nan_indices)))

x1_list = x1_list[not_nan_indices]
x2_list = x2_list[not_nan_indices]
morph_list = morph_list[not_nan_indices]
x1_recon_list = x1_recon_list[not_nan_indices]
x2_recon_list = x2_recon_list[not_nan_indices]

x1_enc = x1_enc[not_nan_indices]
x2_enc = x2_enc[not_nan_indices]
morphs_enc = morphs_enc[not_nan_indices]
x1_recon_enc = x1_recon_enc[not_nan_indices]
x2_recon_enc = x2_recon_enc[not_nan_indices]
x1_inp_enc = x1_inp_enc[not_nan_indices]
x2_inp_enc = x2_inp_enc[not_nan_indices]


# Assert that there are no nans in the comparison faces
# assert not (np.isnan(x1_enc).any() or np.isnan(x2_enc).any())

# Make shifted versions for impostor scores
x1_enc_shifted = np.concatenate([x1_enc[1:], x1_enc[:1]])
x2_enc_shifted = np.concatenate([x2_enc[1:], x2_enc[:1]])


# Compute euclidean distances between x1 and the morph and x2 and the morph
dist_x1 = np.sqrt(np.sum(np.square(x1_enc - morphs_enc), axis=1))
dist_x2 = np.sqrt(np.sum(np.square(x2_enc - morphs_enc), axis=1))
dist_x1_x2 = np.sqrt(np.sum(np.square(x1_enc - x2_enc), axis=1))
dist_x1_recon = np.sqrt(np.sum(np.square(x1_enc - x1_recon_enc), axis=1))
dist_x2_recon = np.sqrt(np.sum(np.square(x2_enc - x2_recon_enc), axis=1))
dist_x1_to_ref = np.sqrt(np.sum(np.square(x1_enc - x1_inp_enc), axis=1))
dist_x2_to_ref = np.sqrt(np.sum(np.square(x2_enc - x2_inp_enc), axis=1))
dist_x1_to_other = np.sqrt(np.sum(np.square(x1_enc - x1_enc_shifted), axis=1))
dist_x2_to_other = np.sqrt(np.sum(np.square(x2_enc - x2_enc_shifted), axis=1))

dist_recon = np.concatenate([dist_x1_recon, dist_x2_recon], axis=0)
dist_ref = np.concatenate([dist_x1_to_ref, dist_x2_to_ref], axis=0)
dist_morph = np.concatenate([dist_x1, dist_x2], axis=0)
dist_mated_impostor = dist_x1_x2
dist_random_impostor = np.concatenate([dist_x1_to_other, dist_x2_to_other], axis=0)


# Replace nan values with the maximal euclidean distance on a 1-D hypersphere (this might not be the correct assumption!)
dist_x1[np.isnan(dist_x1)] = 2.0
dist_x2[np.isnan(dist_x2)] = 2.0
dist_recon[np.isnan(dist_recon)] = 1.0


s = np.stack([dist_x1, dist_x2], axis=1)

mmpmr_value = mmpmr(s, threshold=0.6)
rmd, rmd_values = relative_morph_distance(dist_x1, dist_x2, dist_x1_x2)
correct_reconstruction_rate = (dist_recon < 0.6).mean()

print("Done.")
print()


out_str = ""
out_str += "===== RESULTS =====\n"
out_str += "\n"
out_str += "Computed MMPMR: %s\n"%str(mmpmr_value)
out_str += "Computed reconstruction rate: %s\n"%str(correct_reconstruction_rate)
out_str += "Computed Mean RMD: %s\n"%str(rmd)
out_str += "Mean distance morph to x1 and morph to x2: %s\n"%str(np.concatenate((dist_x1, dist_x2), axis=0).mean())
out_str += "Mean distance x to x_recon: %s\n"%str(dist_recon.mean())
out_str += "\n"
out_str += "==================="

print(out_str)
with open(os.path.join(output_path, "results.txt"), "w") as f:
    f.write(out_str)

json_info = {
    "path": param_path,
    "mmpmr": mmpmr_value,
    "rr": correct_reconstruction_rate,
    "mmd": float(np.concatenate((dist_x1, dist_x2), axis=0).mean()),
    "mrd": float(dist_recon.mean())
}

with open(os.path.join(output_path, "results.json"), "w") as f:
    json.dump(json_info, f)

json_distances = {
    "x_ref_to_x": list(dist_ref),
    "x_ref_to_morph": list(dist_morph),
    "x1_ref_to_x2_ref": list(dist_mated_impostor),
    "x_ref_to_x_recon": list(dist_recon),
    "x_ref_to_random_ref": list(dist_random_impostor)
}
with open(os.path.join(output_path, "distances.json"), "w") as f:
    json.dump(json_distances, f)

if args.visualize:
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.titlesize': 8})

    f = plt.figure(dpi=250)
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
    plt.savefig(os.path.join(output_path, "top5s.png"))
    plt.clf()

    plt.hist(max_distances, bins=60, range=(0.0, 1.2))
    plt.title("Max euclidean distances")
    plt.savefig(os.path.join(output_path, "max_morph_distances.png"))
    plt.clf()

    plt.hist(dist_recon, bins=60, range=(0.0, 1.2))
    plt.title("Reconstruction distances")
    plt.savefig(os.path.join(output_path, "recon_distances.png"))
    plt.clf()

    plt.hist(rmd_values, bins=60)
    plt.title("RMD Values")
    plt.savefig(os.path.join(output_path, "rmd.png"))
    plt.clf()

    all_dists = np.concatenate([dist_ref, dist_mated_impostor, dist_morph, dist_random_impostor])
    all_dists = all_dists[~np.isnan(all_dists)]
    dmin, dmax = all_dists.min(), all_dists.max()
    binwidth = 0.025
    bins = np.arange(dmin, dmax + binwidth, binwidth)
    # plt.hist(dist_recon, bins=60, label="$x^{ref}$ to $x^{recon}$", alpha=0.7, density=True)
    plt.hist(dist_ref, bins=bins, label="genuine", alpha=0.7, density=True, color="blue")
    plt.hist(dist_mated_impostor, bins=bins, label="mated impostor", alpha=0.7, density=True, color="orange")
    plt.hist(dist_random_impostor, bins=bins, label="random impostor", alpha=0.7, density=True, color="red")
    plt.hist(dist_morph, bins=bins, label="morph", alpha=0.7, density=True, color="green")
    plt.ylabel("Density")
    plt.xlabel("FRS encoding euclidean distance")
    plt.legend()
    plt.savefig(os.path.join(output_path, "overview.png"))
    plt.clf()

    sns.distplot(dist_mated_impostor, bins=bins, label="mated impostor", hist=False,  kde=True, color="orange", kde_kws={'shade': True, 'linewidth': 1})
    sns.distplot(dist_morph, bins=bins, label="morph", color="green", hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 1})
    sns.distplot(dist_random_impostor, bins=bins, label="random impostor", hist=False, kde=True, color="red", kde_kws={'shade': True, 'linewidth': 1})
    sns.distplot(dist_ref, bins=bins, label="genuine", hist=False, kde=True, color="blue", kde_kws={'shade': True, 'linewidth': 1})

    plt.ylabel("Density")
    plt.xlabel("FRS encoding euclidean distance")
    plt.legend()
    plt.savefig(os.path.join(output_path, "overview_kernel_density.png"))
    plt.clf()

    plt.hist(
        [dist_ref, dist_mated_impostor, dist_morph, dist_random_impostor],
        label=["$x$ to $x^{ref}$", "$x_1^{ref}$ to $x_2^{ref}$", "$x^{ref}$ to $x^{morph}$", "$x^{ref}$ to random other $x^{ref}$"],
        color=["blue", "orange", "green", "red"],
        bins=40,
        density=True
    )
    plt.ylabel("Density")
    plt.xlabel("FRS encoding euclidean distance")
    plt.legend()
    plt.savefig(os.path.join(output_path, "overview_split_bars.png"))