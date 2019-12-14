"""
    This program loads a model and can be used to do a visual inspection of the results,
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid, save_image

from data.celeba_cropped_pairs_look_alike import CelebaCroppedPairsLookAlike
from models.morphing_encoder import MorphingEncoder
from util.interpolation import slerp
from util.output import init_experiment_output_dir

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
parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the dataset randomly")
parser.add_argument("--use_z_mean", action="store_true", default=False,
                    help="Uses z = z_mean instead of sampling from q(z|x)"
                    )
parser.add_argument("--cuda", action="store_true", default=False, help="When this flag is present, cuda is used")
parser.add_argument("--eval", action="store_true", default=False,
                    help="When this flag is present, the models are put in evaluation mode. This affects BatchNorm")
parser.add_argument("--train", action="store_true", default=False,
                    help="When this flag is present, the models are put in train mode. This affects BatchNorm")
parser.add_argument("--slerp", action="store_true", default=False,
                    help="When this flag is present, slerp interpolation will be used. "
                         "This overrides any other morphing method")
args = parser.parse_args()

output_dir = init_experiment_output_dir("morphing_evaluation", "morphing_evaluation", args)

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
loader = DataLoader(dataset, args.batch_size, shuffle=args.shuffle)

x1_list = []
morph_list = []
x2_list = []

for i, batch in enumerate(loader):
    if args.max_output_batches is not None and i >= args.max_output_batches:
        break
    ((x1, x2), _), idents = batch
    x1_list.append(x1.detach())
    x2_list.append(x2.detach())

    if args.cuda:
        x1 = x1.cuda()
        x2 = x2.cuda()

    if args.slerp:
        z1, z1m, _ = Gz(x1)
        z2, z2m, _ = Gz(x2)

        if args.use_z_mean:
            z1, z2 = z1m, z2m

        z1 = z1.cpu().detach().numpy()
        z2 = z2.cpu().detach().numpy()
        z_morph = np.zeros(z1.shape, dtype=np.float32)
        for j in range(z1.shape[0]):
            z_morph[j] = slerp(0.5, z1[j], z2[j])
        z_morph = torch.from_numpy(z_morph)
        if args.cuda:
            z_morph = z_morph.cuda()
    else:
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
    morph_list.append(x_morph.detach())

x1_list = torch.cat(x1_list, dim=0)
x2_list = torch.cat(x2_list, dim=0)
morph_list = torch.cat(morph_list, dim=0)

x1_column = make_grid(x1_list, nrow=1)
x2_column = make_grid(x2_list, nrow=1)
morph_column = make_grid(morph_list, nrow=1)

if args.tanh:
    out_img = make_grid(torch.stack([x1_column, morph_column, x2_column], dim=0), nrow=3, range=(-1, 1), normalize=True)
else:
    out_img = make_grid(torch.stack([x1_column, morph_column, x2_column], dim=0), nrow=3)
print(out_img.size())
print(os.path.join(output_dir, "morphs.png"))

save_image(out_img, os.path.join(output_dir, "morphs.png"))