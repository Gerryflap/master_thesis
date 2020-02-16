from models.stylegan2.stylegan2_like_discriminator import DeepDiscriminator
from models.stylegan2.stylegan2_like_encoder import DeepEncoder
from models.stylegan2.stylegan2_like_generator import DeepGenerator
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.wgangp_train_loop import GanTrainLoop
from data.celeba_cropped import CelebaCropped
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
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--d_steps", action="store", type=int, default=2,
                    help="Amount of discriminator steps per generator step")
parser.add_argument("--l_size", action="store", type=int, default=12, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--train_enc", action="store_true", default=False, help="Trains an encoder to reconstruct the input")
parser.add_argument("--use_lr_norm_in_D", action="store_true", default=False,
                    help="Use local response norm in D")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba32", "wgan_gp", args)

dataset = CelebaCropped(split="train", download=True, transform=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", download=True, transform=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

G = DeepGenerator(args.l_size, args.h_size, 4, 3, lrn=True)
D = DeepDiscriminator(args.h_size, 32, 3, bn=False, lrn=args.use_lr_norm_in_D)
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
if args.train_enc:
    E = DeepEncoder(args.l_size, args.h_size, 32, 3, lrn=True)
    E_optimizer = torch.optim.Adam(E.parameters(), lr=args.lr, betas=(0.5, 0.999))
else:
    E = None
    E_optimizer = None

if args.cuda:
    G = G.cuda()
    D = D.cuda()
    if E is not None:
        E = E.cuda()

G.init_weights()
D.init_weights()
if E is not None:
    E = E.cuda()

listeners = [
    LossReporter(),
    GanImageSampleLogger(output_path, args, pad_value=1),
    ModelSaver(output_path, n=5, overwrite=True, print_output=True)
]

if E is not None:
    listeners.append(
        AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True)
    )

train_loop = GanTrainLoop(listeners, G, D, G_optimizer, D_optimizer, dataloader, D_steps_per_G_step=args.d_steps,
                          cuda=args.cuda, epochs=args.epochs, E=E, E_optimizer=E_optimizer)

train_loop.train()