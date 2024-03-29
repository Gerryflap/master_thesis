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

parser = argparse.ArgumentParser(description="Mixture ALI experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.003,
                    help="Changes the learning rate, default is 0.003")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--d_h_size", action="store", type=int, default=None,
                    help="Overrides the h_size for D when specified")
parser.add_argument("--epochs", action="store", type=int, default=101, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=2, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--morgan_alpha", action="store", type=float, default=0.3, help="MorGAN alpha")
parser.add_argument("--instance_noise_std", action="store", type=float, default=0.0,
                    help="Standard deviation of instance noise")
parser.add_argument("--r1_gamma", action="store", default=0.0, type=float,
                    help="If > 0, enables R1 loss which pushes the gradient "
                         "norm to zero for real samples in the discriminator.")
parser.add_argument("--ns_gan", action="store_true", default=False,
                    help="Enables non-saturating G loss")
parser.add_argument("--no_D_limit", action="store_true", default=False,
                    help="Disables the limit placed on training D")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "ali", args)

train = MixtureDataset()
valid = MixtureDataset()

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gx = Generator(args.l_size, args.h_size)
Gz = Encoder(args.l_size, args.h_size)
D = Discriminator(args.l_size, args.h_size if args.d_h_size is None else args.d_h_size, mode="ali", batchnorm=False)


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
        discriminator_output=True,
        cuda=args.cuda,
        sample_reconstructions=True,
        every_n_epochs=10,
        output_latent=True,
        output_grad_norm=True,
        ns_gan=args.ns_gan,
    )
]

trainloop = ALITrainLoop(
    listeners,
    Gz,
    Gx,
    D,
    G_optimizer,
    D_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    morgan_alpha=args.morgan_alpha,
    d_img_noise_std=args.instance_noise_std,
    decrease_noise=True,
    r1_reg_gamma=args.r1_gamma,
    non_saturating_G_loss=args.ns_gan,
    disable_D_limiting=args.no_D_limit
)

trainloop.train()