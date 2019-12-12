from torch.optim import RMSprop

from data.celeba_cropped import CelebaCropped
from data.celeba_cropped_pairs import CelebaCroppedPairs

import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from models.conv64_vaegan.discriminator import VAEGANDiscriminator64
from models.conv64_vaegan.encoder import VAEGANEncoder64
from models.conv64_vaegan.generator import VAEGANGenerator64
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver
from trainloops.morph_ae_train_loop import MorphAETrainLoop


parser = argparse.ArgumentParser(description="Celeba Morph AE experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0003,
                    help="Changes the learning rate, default is 0.0003")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=12, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--use_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D, which currently does not work well")
parser.add_argument("--dropout_rate", action="store", default=0.0, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--real_label_value", action="store", type=float, default=1.0, help="Changes the target label for real samples")


args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "morph_ae", args)

dataset = CelebaCroppedPairs(split="train", download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

valid_dataset = CelebaCropped(split="valid", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img * 2 - 1)
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gz = VAEGANEncoder64(args.l_size, args.h_size,  n_channels=3)
Gx = VAEGANGenerator64(args.l_size, args.h_size, n_channels=3, bias=True)
D = VAEGANDiscriminator64(args.h_size, use_bn=args.use_batchnorm_in_D, n_channels=3, dropout=args.dropout_rate)
Gz_optimizer = RMSprop(Gz.parameters(), lr=args.lr)
Gx_optimizer = RMSprop(Gx.parameters(), lr=args.lr)
D_optimizer = RMSprop(D.parameters(), lr=args.lr)

if args.cuda:
    Gz = Gz.cuda()
    Gx = Gx.cuda()
    D = D.cuda()

Gz.init_weights()
Gx.init_weights()
D.init_weights()

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid"),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    KillSwitchListener(output_path)
]
train_loop = MorphAETrainLoop(
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
    real_label_value=args.real_label_value,

)

train_loop.train()
