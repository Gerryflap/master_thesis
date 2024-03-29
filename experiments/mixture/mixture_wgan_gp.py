import argparse

import torch

import util.output
from data.mixture import MixtureDataset
from models.mixture.generator import Generator
from models.mixture.encoder import Encoder
from models.mixture.discriminator import Discriminator
from trainloops.wgangp_train_loop import GanTrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer

parser = argparse.ArgumentParser(description="Mixture WGAN GP experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--lambd", action="store", type=float, default=10.0,
                    help="Lambda, multiplier for gradient penalty")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--d_steps", action="store", type=int, default=5,
                    help="D steps per G step")
parser.add_argument("--epochs", action="store", type=int, default=101, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=2, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")


args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "wgan_gp", args)

train = MixtureDataset()
valid = MixtureDataset()

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gx = Generator(args.l_size, args.h_size)
D = Discriminator(args.l_size, args.h_size, batchnorm=False)


G_optimizer = torch.optim.Adam(Gx.parameters(), lr=args.lr, betas=(0.0, 0.9))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

if args.cuda:
    Gx = Gx.cuda()
    D = D.cuda()

Gx.init_weights()
# Disabled this init since it seems to make it harder for D
# D.init_weights()

listeners = [
    LossReporter(),
    MixtureVisualizer(
        output_path,
        args.l_size,
        valid,
        output_reproductions=False,
        discriminator_output=True,
        cuda=args.cuda,
        sample_reconstructions=True,
        every_n_epochs=10,
    )
]

trainloop = GanTrainLoop(
    listeners,
    Gx,
    D,
    G_optimizer,
    D_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    D_steps_per_G_step=args.d_steps,
    lambd=args.lambd
)

trainloop.train()