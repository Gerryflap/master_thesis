from trainloops.gan_train_loop import GanTrainLoop
from models.conv28.discriminator import Discriminator28
from models.conv28.generator import Generator28
import data
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.gan_image_sample_logger import GanImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter

# Parse commandline arguments
parser = argparse.ArgumentParser(description="MNIST DCGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--d_steps", action="store", type=int, default=1,
                    help="Amount of discriminator steps per generator step")
parser.add_argument("--l_size", action="store", type=int, default=12, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--use_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D, which currently does not work well")
# parser.add_argument("--load_path", action="store", type=str, default=None,
#                     help="When given, loads models from LOAD_PATH folder")
# parser.add_argument("--save_path", action="store", type=str, default=None,
#                     help="When given, saves models to LOAD_PATH folder after all epochs (or every epoch)")
# parser.add_argument("--save_every_epoch", action="store_true", default=False,
#                     help="When a save path is given, store the model after every epoch instead of only the last")
# parser.add_argument("--img_path", action="store", type=str, default=None,
#                     help="When given, saves samples to the given directory")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mnist", "gan", args)

dataset = data.MNIST("data/downloads/mnist", train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

G = Generator28(args.l_size, args.h_size, args.use_mish)
D = Discriminator28(args.h_size, use_bn=args.use_batchnorm_in_D, use_mish=args.use_mish)


if args.cuda:
    G = G.cuda()
    D = D.cuda()

G.init_weights()
D.init_weights()

G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

listeners = [
    LossReporter(),
    GanImageSampleLogger(output_path, args, pad_value=1)
]
train_loop = GanTrainLoop(listeners, G, D, G_optimizer, D_optimizer, dataloader, D_steps_per_G_step=args.d_steps,
                          cuda=args.cuda, epochs=args.epochs)

train_loop.train()
