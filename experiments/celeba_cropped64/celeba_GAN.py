from trainloops.gan_train_loop import GanTrainLoop
from models.conv64.discriminator import Discriminator64
from models.conv64.generator import Generator64
from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse

# Might be useful: https://jjallaire.github.io/deep-learning-with-r-notebooks/notebooks/8.5-introduction-to-gans.nb.html

# Parse commandline arguments
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.listeners.gan_image_sample_logger import GanImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="Celeba DCGAN experiment.")
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
parser.add_argument("--no_bias_in_G", action="store_true", default=False, help="Disables biases in the Generator")
parser.add_argument("--use_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D, which currently does not work well")
parser.add_argument("--dropout_rate", action="store", default=0.0, type=float,
                    help="Sets the dropout rate in D")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "gan", args)

dataset = CelebaCropped(split="train", download=True, transform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

G = Generator64(args.l_size, args.h_size, args.use_mish, not args.no_bias_in_G, n_channels=3)
D = Discriminator64(args.h_size, use_bn=args.use_batchnorm_in_D, use_mish=args.use_mish, n_channels=3, dropout=args.dropout_rate)
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    G = G.cuda()
    D = D.cuda()

G.init_weights()
D.init_weights()

listeners = [
    LossReporter(),
    GanImageSampleLogger(output_path, args, pad_value=1),
    ModelSaver(output_path, n=5, overwrite=True, print_output=True),
    KillSwitchListener(output_path)
]
train_loop = GanTrainLoop(listeners, G, D, G_optimizer, D_optimizer, dataloader, D_steps_per_G_step=args.d_steps,
                          cuda=args.cuda, epochs=args.epochs)

train_loop.train()
