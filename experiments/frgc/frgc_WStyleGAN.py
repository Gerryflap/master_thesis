from PIL import Image

from data.frgc_cropped import FRGCCropped
from models.stylegan2.stylegan1_skip_generator import StyleGenerator
from models.stylegan2.stylegan2_like_discriminator import DeepDiscriminator
from models.stylegan2.stylegan2_like_encoder import DeepEncoder
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
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

parser = argparse.ArgumentParser(description="Fruit Style WGAN-GP experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.001,
                    help="Changes the learning rate, default is 0.001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=1000, help="Sets the number of training epochs")
parser.add_argument("--d_steps", action="store", type=int, default=5,
                    help="Amount of discriminator steps per generator step")
parser.add_argument("--l_size", action="store", type=int, default=128, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--train_enc", action="store_true", default=False, help="Trains an encoder to reconstruct the input")
parser.add_argument("--r1_gamma", action="store", type=float, default=1.0,
                    help="R1 loss gamma")
parser.add_argument("--r1_steps", action="store", type=int, default=1,
                    help="R1 loss is computed every 'r1_steps' steps")
parser.add_argument("--disl", action="store_true", default=False,
                    help="Use disl loss instead of pixel loss")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("frgc48", "w_stylegan", args)

dataset = FRGCCropped(download=True, transform=transforms.Compose([
    transforms.Resize(48, interpolation=Image.LANCZOS),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

G = StyleGenerator(args.l_size, args.h_size, 3, 4, mapping_depth=4)
D = DeepDiscriminator(args.h_size, 48, 4, bn=False, lrn=False)
G_syn_optimizer = torch.optim.Adam(G.get_synthesis_parameters(), lr=args.lr, betas=(0, 0.9))
G_map_optimizer = torch.optim.Adam(G.get_mapping_parameters(), lr=args.lr/100.0, betas=(0, 0.9))
G_optimizer = MergedOptimizers([G_map_optimizer, G_syn_optimizer])
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0, 0.9))
if args.train_enc:
    E = DeepEncoder(args.l_size, args.h_size, 48, 4, lrn=True)
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
    GanImageSampleLogger(output_path, args, pad_value=1, every_n_epochs=10),
    ModelSaver(output_path, n=5, overwrite=True, print_output=True)
]

if E is not None:
    listeners.append(
        AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train", print_stats=True, every_n_epochs=10)
    )

train_loop = GanTrainLoop(listeners, G, D, G_optimizer, D_optimizer, dataloader, D_steps_per_G_step=args.d_steps,
                          cuda=args.cuda, epochs=args.epochs, E=E, E_optimizer=E_optimizer, dis_l=args.disl)

train_loop.train()
