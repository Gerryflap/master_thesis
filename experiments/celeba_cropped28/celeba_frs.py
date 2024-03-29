"""
    Trains a facial recognition system for celeba_cropped28.
"""
import argparse

import torch
from torchvision import transforms

import util.output
from data.celeba_cropped_triplets import CelebaCroppedTriplets
from models.conv28.frs import FRS28
from trainloops.frs_trainloop import FRSTrainLoop
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.listeners.frs_accuracy_printer import FRSAccuracyPrinter
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="Celeba FRS experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=128, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--n_negatives", action="store", type=int, default=1,
                    help="if > 1, switch to hard triplet selection and select hardest negative from n negatives")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba28", "frs", args)

dataset = CelebaCroppedTriplets(split="train", download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]), give_n_negatives=args.n_negatives)

valid_dataset = CelebaCroppedTriplets(split="valid", download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

frs_model = FRS28(args.l_size, args.h_size, use_mish=args.use_mish, n_channels=3, add_dense_layer=True, hypersphere_output=True)

if args.cuda:
    frs_model = frs_model.cuda()

optimizer = torch.optim.Adam(frs_model.parameters(), lr=args.lr)

frs_model.init_weights()

listeners = [
    LossReporter(),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    FRSAccuracyPrinter(args.cuda, valid_dataset),
    KillSwitchListener(output_path)
]

trainloop = FRSTrainLoop(
    listeners,
    frs_model,
    optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    margin=0.2,
    use_hard_triplets=args.n_negatives > 1
)
trainloop.train()