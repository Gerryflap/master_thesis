from data.celeba_cropped import CelebaCropped
from models.stylegan2.stylegan1_skip_generator import StyleGenerator
from models.stylegan2.stylegan2_like_discriminator import DeepDiscriminator
from models.stylegan2.stylegan2_like_encoder import DeepEncoder
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.wgangp_train_loop import GanTrainLoop
import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from trainloops.listeners.gan_image_sample_logger import GanImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver
from util.misc import MergedOptimizers

parser = argparse.ArgumentParser(description="Celeba Style WGAN-GP experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.001,
                    help="Changes the learning rate, default is 0.001")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=1000, help="Sets the number of training epochs")
parser.add_argument("--d_steps", action="store", type=int, default=1,
                    help="Amount of discriminator steps per generator step")
parser.add_argument("--l_size", action="store", type=int, default=512, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--train_enc", action="store_true", default=False, help="Trains an encoder to reconstruct the input")
parser.add_argument("--disl", action="store_true", default=False,
                    help="Use disl loss instead of pixel loss")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "w_stylegan", args)

dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

G = StyleGenerator(args.l_size, args.h_size, 4, 4, mapping_depth=8)
D = DeepDiscriminator(args.h_size, 64, 4, bn=False, lrn=False)
G_syn_optimizer = torch.optim.Adam(G.get_synthesis_parameters(), lr=args.lr, betas=(0, 0.9))
G_map_optimizer = torch.optim.Adam(G.get_mapping_parameters(), lr=args.lr/100.0, betas=(0, 0.9))
G_optimizer = MergedOptimizers([G_map_optimizer, G_syn_optimizer])
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0, 0.9))
if args.train_enc:
    E = DeepEncoder(args.l_size, args.h_size, 64, 4, lrn=True)
    E_optimizer = torch.optim.Adam(E.parameters(), lr=args.lr/5, betas=(0.5, 0.999))
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
    E.init_weights()

listeners = [
    LossReporter(),
    GanImageSampleLogger(output_path, args, pad_value=1, every_n_epochs=1),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    ModelSaver(output_path, n=50, overwrite=False, print_output=True),
    KillSwitchListener(output_path)
]

if E is not None:
    listeners.append(
        AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train", print_stats=False, every_n_epochs=1)
    )
    listeners.append(
        AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True, every_n_epochs=1)
    )

train_loop = GanTrainLoop(listeners, G, D, G_optimizer, D_optimizer, dataloader, D_steps_per_G_step=args.d_steps,
                          cuda=args.cuda, epochs=args.epochs, E=E, E_optimizer=E_optimizer, dis_l=args.disl)

train_loop.train()
