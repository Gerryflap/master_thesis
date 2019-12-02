import argparse

import torch

import util.output
from data.mixture import MixtureDataset
from models.mixture.generator import Generator
from models.mixture.encoder import Encoder
from models.mixture.discriminator import Discriminator
from trainloops.veegan_train_loop import VEEGANTrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer

parser = argparse.ArgumentParser(description="Mixture VEEGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.003,
                    help="Changes the learning rate, default is 0.003")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=101, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=2, help="Size of the latent space")
parser.add_argument("--pre_train_steps", action="store", type=int, default=0, help="Number of pre training steps for Gz")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--extended_reproduction_step", action="store_true", default=False,
                    help="Adds a reconstruction loss between Gz(x) and Gz(Gx(Gz(x))).")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "veegan", args)

train = MixtureDataset(datapoints_per_grid_position=5)
valid = MixtureDataset(datapoints_per_grid_position=5)

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gx = Generator(args.l_size, args.h_size)
Gz = Encoder(args.l_size, args.h_size, deterministic=True)
D = Discriminator(args.l_size, args.h_size, mode="ali")


G_optimizer = torch.optim.Adam(list(Gz.parameters()) + list(Gx.parameters()), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    Gz = Gz.cuda()
    Gx = Gx.cuda()
    D = D.cuda()

Gz.init_weights()
Gx.init_weights()
D.init_weights()

listeners = [
    LossReporter(),
    MixtureVisualizer(
        output_path,
        args.l_size,
        valid,
        output_reproductions=True,
        discriminator_output=False,
        cuda=args.cuda,
        sample_reconstructions=True,
        every_n_epochs=200
    )
]

trainloop = VEEGANTrainLoop(
    listeners,
    Gz,
    Gx,
    D,
    G_optimizer,
    D_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    pre_training_steps=args.pre_train_steps,
    extended_reproduction_step=args.extended_reproduction_step
)

trainloop.train()