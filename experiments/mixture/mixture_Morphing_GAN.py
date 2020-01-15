import argparse

import torch

import util.output
from data.mixture import MixtureDataset
from data.mixture_pairs import MixturePairs
from models.mixture.generator import Generator
from models.mixture.encoder import Encoder
from models.mixture.discriminator import Discriminator
from trainloops.ali_train_loop import ALITrainLoop
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.mixture_visualizer import MixtureVisualizer
from trainloops.morphing_gan_train_loop import MorphingGANTrainLoop

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
parser.add_argument("--morgan_alpha", action="store", type=float, default=0.3, help="MorGAN alpha")
parser.add_argument("--instance_noise_std", action="store", type=float, default=0.1,
                    help="Standard deviation of instance noise")
parser.add_argument("--morph_loss_factor", action="store", default=0.3, type=float,
                    help="Scales the morph loss")
parser.add_argument("--use_dis_l_reconstruction_loss", action="store_true", default=False,
                    help="Switches the reconstruction loss to a VAEGAN like loss instead of pixelwise.")
parser.add_argument("--use_dis_l_morph_loss", action="store_true", default=False,
                    help="Switches the morph loss to a VAEGAN like loss instead of pixelwise.")


args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mixture", "Morphing_GAN", args)

train = MixturePairs()
valid = MixturePairs()

dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gx = Generator(args.l_size, args.h_size)
Gz = Encoder(args.l_size, args.h_size)
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
        every_n_epochs=10
    )
]

trainloop = MorphingGANTrainLoop(
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
    morph_loss_factor=args.morph_loss_factor,
    reconstruction_loss_mode="pixelwise" if not args.use_dis_l_reconstruction_loss else "dis_l",
    morph_loss_mode="pixelwise" if not args.use_dis_l_morph_loss else "dis_l"
)

trainloop.train()