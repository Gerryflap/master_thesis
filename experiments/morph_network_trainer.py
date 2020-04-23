import argparse
import os

from torchvision.transforms import transforms

import util.output

import torch

from data.celeba_cropped_pairs import CelebaCroppedPairs
from models.morph.morph_network import MorphNetwork
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver
from trainloops.listeners.morph_image_logger import MorphImageLogger
from trainloops.morph_network_train_loop import MorphNetTrainLoop

parser = argparse.ArgumentParser(description="Celeba Morph network trainer.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--morph_loss_factor", action="store", default=1.0, type=float,
                    help="Scales the morph loss wrt. the regularization terms")
parser.add_argument("--use_dis_l_morph_loss", action="store_true", default=False,
                    help="Switches the morph loss to a VAEGAN like loss instead of pixelwise.")
parser.add_argument("--trained_net_path", action="store", required=True, help="Path to the folder containing the "
                                                                             "trained Gx, Gz and D")
parser.add_argument("--res", action="store", type=int, default=64, help="Sets the resolution")
parser.add_argument("--decoder_filename", action="store", type=str, default="Gx.pt",
                    help="Filename of the decoder/generator/Gx network. "
                         "Usually this option can be left at the default value, which is Gx.pt")
parser.add_argument("--encoder_filename", action="store", type=str, default="Gz.pt",
                    help="Filename of the encoder/Gz network. "
                         "Usually this option can be left at the default value, which is Gz.pt")
parser.add_argument("--discriminator_filename", action="store", type=str, default="D.pt",
                    help="Filename of the discriminator network. Only used with gradient descend morphing. "
                         "Default is D.pt")
parser.add_argument("--use_two_sided_z_l2_loss", action="store_true", default=False,
                    help="Uses the mse between the mean l2 norm of z1 and z2 and the l2 norm of z_morph "
                         "as a regularization instead of the default.")
args = parser.parse_args()

param_path = args.trained_net_path
device = torch.device("cpu") if not args.cuda else torch.device("cuda")
Gx = torch.load(os.path.join(param_path, args.decoder_filename), map_location=device)
Gz = torch.load(os.path.join(param_path, args.encoder_filename), map_location=device)
D = torch.load(os.path.join(param_path, args.discriminator_filename), map_location=device)

output_path = util.output.init_experiment_output_dir("celeba%d"%args.res, "MorphNet", args)

dataset = CelebaCroppedPairs(split="train", download=True, transform=transforms.Compose([
    transforms.Resize(args.res),
    transforms.ToTensor(),
]))

valid_dataset = CelebaCroppedPairs(split="valid", download=True, transform=transforms.Compose([
    transforms.Resize(args.res),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

latent_size = Gx.latent_size

morph_net = MorphNetwork(latent_size)
if args.cuda:
    morph_net = morph_net.cuda()
morph_net.init_weights()

optim = torch.optim.Adam(morph_net.parameters(), lr=args.lr)
morph_net.pretrain_morph_network(optim)

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    MorphImageLogger(output_path, valid_dataset, args, slerp=False),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    ModelSaver(output_path, n=30, overwrite=False, print_output=True),
    KillSwitchListener(output_path)
]

if args.use_dis_l_morph_loss:
    morph_loss = "dis_l"
else:
    morph_loss = "pixelwise"

trainloop = MorphNetTrainLoop(
    listeners=listeners,
    morph_net=morph_net,
    Gz=Gz,
    Gx=Gx,
    D=D,
    optim=optim,
    dataloader=dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    morph_loss_factor=args.morph_loss_factor,
    morph_loss_mode=morph_loss,
    two_sided_l2_constraint=args.use_two_sided_z_l2_loss
)

trainloop.train()
