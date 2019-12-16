import argparse

import torch

import util.output
from data.mixture import MixtureDataset
from models.mixture.generator import Generator
from models.mixture.discriminator import Discriminator
from trainloops.gan_train_loop import GanTrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer

parser = argparse.ArgumentParser(description="Mixture GAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.003")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=101, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=2, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")


args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "gan", args)

train = MixtureDataset(stddev=0, hrange=(0, 1), vrange=(0, 1))
valid = MixtureDataset(stddev=0, hrange=(0, 1), vrange=(0, 1))

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

G = Generator(args.l_size, args.h_size)
D = Discriminator(args.l_size, args.h_size)


G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    G = G.cuda()
    D = D.cuda()

G.init_weights()
D.init_weights()

listeners = [
    LossReporter(),
    MixtureVisualizer(
        output_path,
        args.l_size,
        valid,
        output_reproductions=False,
        discriminator_output=True,
        cuda=args.cuda,
        sample_reconstructions=False,
        every_n_epochs=10,
        output_latent=False
    )
]

trainloop = GanTrainLoop(
    listeners,
    G,
    D,
    G_optimizer,
    D_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
)

trainloop.train()