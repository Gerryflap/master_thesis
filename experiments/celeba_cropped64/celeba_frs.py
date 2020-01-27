"""
    Trains a facial recognition system for celeba_cropped28.
"""
import argparse

import torch
from torchvision import transforms

import util.output
from data.celeba_cropped_triplets import CelebaCroppedTriplets
from models.conv64.frs import FRS64
from trainloops.frs_trainloop import FRSTrainLoop
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.listeners.frs_accuracy_printer import FRSAccuracyPrinter
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="Celeba FRS experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=64,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=128, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "frs", args)

dataset = CelebaCroppedTriplets(split="train", download=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

valid_dataset = CelebaCroppedTriplets(split="valid", download=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

frs_model = FRS64(args.l_size, args.h_size, use_mish=args.use_mish, n_channels=3)

if args.cuda:
    frs_model = frs_model.cuda()

optimizer = torch.optim.Adam(frs_model.parameters(), lr=args.lr)

frs_model.init_weights()

listeners = [
    LossReporter(),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    FRSAccuracyPrinter(args.cuda, valid_dataset, margin=1.0),
    KillSwitchListener(output_path)
]

trainloop = FRSTrainLoop(
    listeners,
    frs_model,
    optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    margin=1.0
)
trainloop.train()