from data.celeba_cropped import CelebaCropped
from models.conv64.discriminator import Discriminator64
from models.conv64_ali.generator import Generator64
from models.stylegan2.stylegan2_like_discriminator import DeepDiscriminator
from models.stylegan2.stylegan2_like_generator import DeepGenerator
from trainloops.wgangp_train_loop import GanTrainLoop
import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from trainloops.listeners.gan_image_sample_logger import GanImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="Celeba Deep WGAN-GP experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=64,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--d_steps", action="store", type=int, default=5,
                    help="Amount of discriminator steps per generator step")
parser.add_argument("--l_size", action="store", type=int, default=32, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--dropout_rate", action="store", default=0.0, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--use_lr_norm", action="store_true", default=False,
                    help="Uses local response norm, which will make the generator samples "
                         "independent from the rest of the batch.")
parser.add_argument("--lambd", action="store", type=float, default=10.0,
                    help="Lambda, multiplier for gradient penalty")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "deep_wgan_gp", args)

dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))



dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

G = DeepGenerator(args.l_size, args.h_size, 4, 4)
D = DeepDiscriminator(args.h_size, 64, 4, 3)

G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

if args.cuda:
    G = G.cuda()
    D = D.cuda()

listeners = [
    LossReporter(),
    GanImageSampleLogger(output_path, args, pad_value=1, n_images=6*6),
    ModelSaver(output_path, n=5, overwrite=True, print_output=True),
    ModelSaver(output_path, n=20, overwrite=False, print_output=True)

]
train_loop = GanTrainLoop(listeners, G, D, G_optimizer, D_optimizer, dataloader, D_steps_per_G_step=args.d_steps,
                          cuda=args.cuda, epochs=args.epochs, lambd=args.lambd)

train_loop.train()
