"""
    This file generates a plot of L_recon wrt the two input images when interpolating between them for the validation set
    , given a model.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from data.celeba_cropped_pairs_look_alike import CelebaCroppedPairsLookAlike
import matplotlib.pyplot as plt
import numpy as np

from util.torch.losses import euclidean_distance, euclidean_distance_per_element

parser = argparse.ArgumentParser(description="Morph Loss Plotter.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Image resolution")
parser.add_argument("--param_path", action="store", type=str, required=True, help="Path to E/G/D model folder")
parser.add_argument("--n_steps", action="store", type=int, default=11, help="Number of in-between points to pick")
parser.add_argument("--res", action="store", type=int, default=64, help="Image resolution")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="If true, a sample is taken from the encoder output distribution. Otherwise the mean is used.")
parser.add_argument("--use_dis_l", action="store_true", default=False,
                    help="Switches to Dis_l loss instead of pixels")
parser.add_argument("--n_batches", action="store", type=int, default=None, help="Number of batches to process")
parser.add_argument("--frs_path", action="store", default=None, help="Path to facial recognition system model. "
                                                                     "Switches to FRS reconstruction loss")
args = parser.parse_args()

trans = []
if args.res != 64:
    trans.append(transforms.Resize(args.res))

trans.append(transforms.ToTensor())
dataset = CelebaCroppedPairsLookAlike(split="valid", transform=transforms.Compose(trans))
dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

fname_gz = os.path.join(args.param_path, "Gz.pt")
Gz = torch.load(fname_gz, map_location=torch.device('cpu'))
Gz.eval()

fname_gx = os.path.join(args.param_path, "Gx.pt")
Gx = torch.load(fname_gx, map_location=torch.device('cpu'))
Gx.eval()

# Code for loading frs model when frs based reconstruction loss is used
frs_model = None
if args.frs_path is not None:
    frs_model = torch.load(args.frs_path)
    frs_model.requires_grad = False
    frs_model.eval()
    if args.cuda:
        frs_model = frs_model.cuda()

D = None
if args.use_dis_l:
    fname_D = os.path.join(args.param_path, "D.pt")

    D = torch.load(fname_D, map_location=torch.device('cpu'))
    D.eval()

if args.cuda:
    if D is not None:
        D = D.cuda()
    Gx = Gx.cuda()
    Gz = Gz.cuda()


def disl_losses(pred, target1, target2):
    _, disl_pred = D.compute_dx(pred)
    _, disl_target1 = D.compute_dx(target1)
    _, disl_target2 = D.compute_dx(target2)
    loss1 = torch.nn.functional.mse_loss(disl_pred, disl_target1, reduce=False).mean(dim=(1, 2, 3))
    loss2 = torch.nn.functional.mse_loss(disl_pred, disl_target2, reduce=False).mean(dim=(1, 2, 3))
    return loss1, loss2

def frs_losses(pred, target1, target2):
    emb_pred = frs_model(pred)
    emb_target1 = frs_model(target1)
    emb_target2 = frs_model(target2)
    loss1 = euclidean_distance_per_element(emb_pred, emb_target1)
    loss2 = euclidean_distance_per_element(emb_pred, emb_target2)
    return loss1, loss2


x1_losses = []
x2_losses = []

for i, batch in enumerate(dataloader):
    if args.n_batches is not None and i == args.n_batches:
        break
    ((x1, x2), (x1_comp, x2_comp)), idents = batch

    if args.cuda:
        x1 = x1.cuda()
        x2 = x2.cuda()

    z1 = Gz.encode(x1)
    z2 = Gz.encode(x2)

    x1_batch_losses = []
    x2_batch_losses = []

    for i in range(args.n_steps):
        z2_amount = float(i)/(args.n_steps - 1)
        z_morph = z2 * z2_amount + z1 * (1.0-z2_amount)
        x_morph = Gx(z_morph)

        if args.use_dis_l:
            loss_1, loss_2 = disl_losses(x_morph, x1, x2)
        elif frs_model is not None:
            loss_1, loss_2 = frs_losses(x_morph, x1, x2)
        else:
            loss_1 = torch.nn.functional.l1_loss(x_morph, x1, reduce=False).mean(dim=(1, 2, 3))
            loss_2 = torch.nn.functional.l1_loss(x_morph, x2, reduce=False).mean(dim=(1, 2, 3))
        x1_batch_losses.append(loss_1.detach())
        x2_batch_losses.append(loss_2.detach())
    x1_batch_losses = torch.stack(x1_batch_losses, dim=0)
    x2_batch_losses = torch.stack(x2_batch_losses, dim=0)
    x1_losses.append(x1_batch_losses)
    x2_losses.append(x2_batch_losses)

x1_losses = torch.cat(x1_losses, dim=1).cpu()
x2_losses = torch.cat(x2_losses, dim=1).cpu()

steps = np.array([i/(args.n_steps - 1) for i in range(args.n_steps)])

# Loop over batch dim
for i in range(x1_losses.size(1)):
    plt.plot(steps, x1_losses[:, i].numpy(), color="blue", linewidth=1, alpha=0.3)
    plt.plot(steps, x2_losses[:, i].numpy(), color="red", linewidth=1, alpha=0.3)

plt.plot(
    [0.5, 0.5],
    [
        max(x1_losses.max(), x2_losses.max()),
        min(x1_losses.min(), x2_losses.min())
    ],
    color="orange"
)
plt.ylabel("Loss")
plt.xlabel("Amount of z2 in z_morph.")
plt.title("All loss graphs")
plt.show()

x1_losses_mean = x1_losses.mean(dim=1).numpy()
x2_losses_mean = x2_losses.mean(dim=1).numpy()
plt.plot(steps, x1_losses_mean, color="blue")
plt.plot(steps, x2_losses_mean, color="red")
plt.plot(
    [0.5, 0.5],
    [
        max(x1_losses_mean.max(), x2_losses_mean.max()),
        min(x1_losses_mean.min(), x2_losses_mean.min())
    ],
    color="orange"
)
plt.ylabel("Mean loss")
plt.xlabel("Amount of z2 in z_morph.")
plt.title("Mean loss graph over dataset")
plt.show()





