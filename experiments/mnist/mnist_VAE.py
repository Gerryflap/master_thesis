from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.vae_train_loop import VaeTrainLoop
from models.conv28.encoder import Encoder28
from models.conv28.generator import Generator28
import data
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.loss_reporter import LossReporter

# Parse commandline arguments
parser = argparse.ArgumentParser(description="MNIST VAE experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--d_steps", action="store", type=int, default=2,
                    help="Amount of discriminator steps per generator step")
parser.add_argument("--l_size", action="store", type=int, default=12, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--no_bias_in_dec", action="store_true", default=False, help="Disables biases in the decoder")
# parser.add_argument("--load_path", action="store", type=str, default=None,
#                     help="When given, loads models from LOAD_PATH folder")
# parser.add_argument("--save_path", action="store", type=str, default=None,
#                     help="When given, saves models to LOAD_PATH folder after all epochs (or every epoch)")
# parser.add_argument("--save_every_epoch", action="store_true", default=False,
#                     help="When a save path is given, store the model after every epoch instead of only the last")
# parser.add_argument("--img_path", action="store", type=str, default=None,
#                     help="When given, saves samples to the given directory")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mnist", "vae", args)

dataset = data.MNIST("data/downloads/mnist", train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

valid_dataset = data.MNIST("data/downloads/mnist", train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


enc = Encoder28(args.l_size, args.h_size, args.use_mish)
dec = Generator28(args.l_size, args.h_size, args.use_mish, not args.no_bias_in_dec)
enc_optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr, betas=(0.5, 0.999))
dec_optimizer = torch.optim.Adam(dec.parameters(), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    enc = enc.cuda()
    dec = dec.cuda()

enc.init_weights()
dec.init_weights()

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args),
]
train_loop = VaeTrainLoop(listeners, enc, dec, enc_optimizer, dec_optimizer, dataloader,
                          cuda=args.cuda, epochs=args.epochs)

train_loop.train()
