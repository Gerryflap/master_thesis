import argparse

import torch

import util.output
from data.mixture import MixtureDataset
from models.mixture.generator import Generator
from models.mixture.encoder import Encoder
from models.mixture.discriminator import Discriminator
from trainloops.ali_train_loop import ALITrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer
from trainloops.wcyclegan_train_loop import WCycleGanTrainLoop

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
parser.add_argument("--alpha", action="store", type=float, default=1.0, help="Alpha, scales z reconstruction loss")
parser.add_argument("--d_steps", action="store", type=int, default=5, help="D steps per G step")
parser.add_argument("--lambdx", action="store", type=float, default=0.1,
                    help="Lambda, multiplier for gradient penalty on x")
parser.add_argument("--lambdz", action="store", type=float, default=0.1,
                    help="Lambda, multiplier for gradient penalty on z")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "wcyclegan", args)

train = MixtureDataset()
valid = MixtureDataset()

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gx = Generator(args.l_size, args.h_size)
Gz = Encoder(args.l_size, args.h_size, deterministic=True)
Dx = Discriminator(args.l_size, args.h_size, mode="normal")
Dz = Discriminator(args.l_size, args.h_size, mode="normal", input_size=args.l_size)


G_optimizer = torch.optim.Adam(list(Gz.parameters()) + list(Gx.parameters()), lr=args.lr, betas=(0, 0.9))
D_optimizer = torch.optim.Adam(list(Dx.parameters()) + list(Dz.parameters()), lr=args.lr, betas=(0, 0.9))

if args.cuda:
    Gz = Gz.cuda()
    Gx = Gx.cuda()
    Dx = Dx.cuda()
    Dz = Dz.cuda()

Gz.init_weights()
Gx.init_weights()

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
    )
]

trainloop = WCycleGanTrainLoop(
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
    alpha=args.alpha,
    lambd_x=args.lambdx,
    lambd_z=args.lambdz,
    D_steps_per_G_step=args.d_steps
)

trainloop.train()