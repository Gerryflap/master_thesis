import argparse

import torch

import util.output
from data.mixture import MixtureDataset
from models.mixture.generator import Generator
from models.mixture.encoder import Encoder
from models.mixture.discriminator import Discriminator
from trainloops.MorGAN_dual_discriminator_train_loop import MorGANDDTrainLoop
from trainloops.wgangp_train_loop import GanTrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer

parser = argparse.ArgumentParser(description="Mixture Morgan DD experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--d_steps", action="store", type=int, default=1,
                    help="D steps per G step")
parser.add_argument("--epochs", action="store", type=int, default=101, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=2, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--alpha_x", action="store", type=float, default=0.3,
                    help="x reconstruction loss scaling factor")
parser.add_argument("--alpha_z", action="store", type=float, default=0.3,
                    help="z reconstruction loss scaling factor")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "MorGAN_DD", args)

train = MixtureDataset()
valid = MixtureDataset()

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gx = Generator(args.l_size, args.h_size)
Gz = Encoder(args.l_size, args.h_size)
Dx = Discriminator(args.l_size, args.h_size, batchnorm=True)
Dz = Discriminator(args.l_size, args.h_size, batchnorm=True, input_size=args.l_size)


G_optimizer = torch.optim.Adam(list(Gx.parameters()) + list(Gz.parameters()), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(list(Dx.parameters()) + list(Dz.parameters()), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    Gx = Gx.cuda()
    Dx = Dx.cuda()
    Gz = Gz.cuda()
    Dz = Dz.cuda()

Gx.init_weights()
Gz.init_weights()
Dx.init_weights()
Dz.init_weights()

listeners = [
    LossReporter(),
    MixtureVisualizer(
        output_path,
        args.l_size,
        valid,
        output_reproductions=True,
        discriminator_output=True,
        cuda=args.cuda,
        sample_reconstructions=True,
        every_n_epochs=10,
        output_morph_path=True,
        output_latent=True
    )
]

trainloop = MorGANDDTrainLoop(
    listeners,
    Gz,
    Gx,
    Dz,
    Dx,
    G_optimizer,
    D_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    D_steps_per_G_step=args.d_steps,
    alpha_x=args.alpha_x,
    alpha_z=args.alpha_z
)

trainloop.train()