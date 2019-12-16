from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.encoder_based_discriminator import EncBasedDecTrainLoop
from models.conv28.encoder import Encoder28
from models.conv28.generator import Generator28
import data
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.loss_reporter import LossReporter

# Parse commandline arguments
parser = argparse.ArgumentParser(description="MNIST VAE as discriminator experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--pre_train_epochs", action="store", type=int, default=5, help="Sets the number of pre-training epochs")
parser.add_argument("--l_size", action="store", type=int, default=12, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--beta", action="store", type=float, default=1.0,
                    help="Beta scales the latent space regularization")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mnist", "vae_discr", args)

dataset = data.MNIST("data/downloads/mnist", train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

valid_dataset = data.MNIST("data/downloads/mnist", train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


G = Generator28(args.l_size, args.h_size, bias=False)
enc = Encoder28(args.l_size, args.h_size, deterministic=False, no_bn_in_first_layer=True)
dec = Generator28(args.l_size, args.h_size, bias=False)

G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    G = G.cuda()
    dec = dec.cuda()
    enc = enc.cuda()

G.init_weights()
enc.init_weights()
dec.init_weights()
listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args),
]

train_loop = EncBasedDecTrainLoop(
    listeners,
    G,
    enc,
    dec,
    G_optimizer,
    D_optimizer,
    args.pre_train_epochs,
    dataloader,
    args.cuda,
    args.epochs,
    beta=args.beta,
)


train_loop.train()
