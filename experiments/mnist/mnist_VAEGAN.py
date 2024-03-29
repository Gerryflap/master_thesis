"""
    VEEGAN on the MNIST dataset.
"""

from torchvision.datasets import MNIST

import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from models.conv28.vaegan_discriminator import VAEGANDiscriminator28
from models.conv28_vaegan.encoder import VAEGANEncoder28
from models.conv28_vaegan.generator import VAEGANGenerator28
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver
from trainloops.vae_gan_train_loop import VAEGANTrainLoop

parser = argparse.ArgumentParser(description="MNIST VAEGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--fc_h_size", action="store", type=int, default=None,
                    help="Sets the fc_h_size, which changes the size of the fully connected layers in D")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=12, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--gamma", action="store", type=float, default=1e-6,
                    help="Gamma scales L_disl_llike in the Gx/Decoder loss")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--use_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D, which currently does not work well")
parser.add_argument("--dropout_rate", action="store", default=0.0, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--no_reconstructions_to_D",  action="store_true", default=False,
                    help="When this flag is used, samples from Gx(Gz(x)) will not be fed to D.")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mnist", "vaegan", args)

dataset = MNIST("data/downloads/mnist", train=True, download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

valid_dataset = MNIST("data/downloads/mnist", train=False, download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

Gz = VAEGANEncoder28(args.l_size, args.h_size,  n_channels=1)
Gx = VAEGANGenerator28(args.l_size, args.h_size, n_channels=1)
D = VAEGANDiscriminator28(args.h_size, use_bn=args.use_batchnorm_in_D, use_mish=args.use_mish, n_channels=1, dropout=args.dropout_rate)

if args.cuda:
    Gz = Gz.cuda()
    Gx = Gx.cuda()
    D = D.cuda()

Gz_optimizer = torch.optim.Adam(Gz.parameters(), lr=args.lr, betas=(0.5, 0.999))
Gx_optimizer = torch.optim.Adam(Gx.parameters(), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

Gz.init_weights()
Gx.init_weights()
D.init_weights()

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid"),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid_train_mode", eval_mode=False),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True)
]
train_loop = VAEGANTrainLoop(
    listeners=listeners,
    Gz=Gz,
    Gx=Gx,
    D=D,
    optim_Gz=Gz_optimizer,
    optim_Gx=Gx_optimizer,
    optim_D=D_optimizer,
    dataloader=dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    gamma=args.gamma,
    feed_reconstructions_into_D=not args.no_reconstructions_to_D
)

train_loop.train()
