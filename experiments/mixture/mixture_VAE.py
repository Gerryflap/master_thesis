import argparse

import torch

import util.output
from data.mixture import MixtureDataset
from models.mixture.generator import Generator
from models.mixture.encoder import Encoder
from models.mixture.discriminator import Discriminator
from trainloops.vae_train_loop import VaeTrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer

parser = argparse.ArgumentParser(description="Mixture VAE experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.001,
                    help="Changes the learning rate, default is 0.001")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=101, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=2, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--beta", action="store", type=float, default=1.0,
                    help="Beta. Weight for L_prior")


args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "vae", args)

train = MixtureDataset()
valid = MixtureDataset()

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

enc = Encoder(args.l_size, args.h_size)
dec = Generator(args.l_size, args.h_size)


enc_optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr, betas=(0.5, 0.999))
dec_optimizer = torch.optim.Adam(dec.parameters(), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    enc = enc.cuda()
    dec = dec.cuda()

enc.init_weights()
dec.init_weights()

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
        every_n_epochs=10,
        output_latent=True
    )
]

trainloop = VaeTrainLoop(
    listeners,
    enc,
    dec,
    enc_optimizer,
    dec_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    beta=args.beta
)

trainloop.train()