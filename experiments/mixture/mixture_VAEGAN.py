import argparse

import torch
from torch.optim import RMSprop

import util.output
from data.mixture import MixtureDataset
from models.mixture.generator import Generator
from models.mixture.encoder import Encoder
from models.mixture.discriminator import Discriminator
from trainloops.vae_gan_train_loop import VAEGANTrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer

parser = argparse.ArgumentParser(description="Mixture ALI experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.003,
                    help="Changes the learning rate, default is 0.003")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=101, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=2, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--gamma", action="store", type=float, default=1e-3, help="Changes the gamma used by VAE/GAN")
parser.add_argument("--beta", action="store", type=float, default=1.0, help="Scales L_prior")



args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "vaegan", args)

train = MixtureDataset()
valid = MixtureDataset()

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gx = Generator(args.l_size, args.h_size)
Gz = Encoder(args.l_size, args.h_size)
D = Discriminator(args.l_size, args.h_size, mode="vaegan")

Gz_optimizer = RMSprop(Gz.parameters(), lr=args.lr)
Gx_optimizer = RMSprop(Gx.parameters(), lr=args.lr)
D_optimizer = RMSprop(D.parameters(), lr=args.lr)

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
        every_n_epochs=10
    )
]

trainloop = VAEGANTrainLoop(
    listeners,
    Gz,
    Gx,
    D,
    Gz_optimizer,
    Gx_optimizer,
    D_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    gamma=args.gamma,
    beta=args.beta
)

trainloop.train()