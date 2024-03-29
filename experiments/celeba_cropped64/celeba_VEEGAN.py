import os

from models.conv64_ali.ali_discriminator import ALIDiscriminator64
from models.conv64_ali.encoder import Encoder64
from models.conv64_ali.veegan_discriminator import VEEGANDiscriminator64
from models.conv64_ali.generator import Generator64

from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver
from trainloops.veegan_train_loop import VEEGANTrainLoop

from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments


parser = argparse.ArgumentParser(description="Celeba ALI experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.00001,
                    help="Changes the learning rate, default is 0.00001")
parser.add_argument("--h_size", action="store", type=int, default=64,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--fc_h_size", action="store", type=int, default=None,
                    help="Sets the fc_h_size, which changes the size of the fully connected layers in D")
parser.add_argument("--epochs", action="store", type=int, default=123, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=256, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--disable_batchnorm_in_D", action="store_true", default=False,
                    help="Disables batch normalization in D")
parser.add_argument("--dropout_rate", action="store", default=0.2, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--continue_with", action="store", type=str, default=None,
                    help="Path the the experiment to load. Keep hyperparams the same!")
parser.add_argument("--instance_noise_std", action="store", default=0.0, type=float,
                    help="Sets the standard deviation for instance noise (noise added to inputs of D)")
parser.add_argument("--d_real_label", action="store", default=1.0, type=float,
                    help="Changes the label value for the \"real\" output of D. "
                         "This can be used for label smoothing. "
                         "Recommended is 1.0 for no smoothing or 0.9 for smoothing")
parser.add_argument("--pre_train_steps", action="store", type=int, default=0, help="Number of pre training steps for Gz")


args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "VEEGAN", args)


dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.continue_with is None:
    Gz = Encoder64(args.l_size, args.h_size, args.use_mish, n_channels=3)
    Gx = Generator64(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True)
    D = ALIDiscriminator64(args.l_size, args.h_size, use_bn=not args.disable_batchnorm_in_D, use_mish=args.use_mish,
                           n_channels=3, dropout=args.dropout_rate, fc_h_size=args.fc_h_size)
    G_optimizer = torch.optim.Adam(list(Gz.parameters()) + list(Gx.parameters()), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

else:
    Gz = torch.load(os.path.join(args.continue_with, "params", "all_epochs", "Gz.pt"), map_location=torch.device('cpu'))
    Gx = torch.load(os.path.join(args.continue_with, "params", "all_epochs", "Gx.pt"), map_location=torch.device('cpu'))
    D = torch.load(os.path.join(args.continue_with, "params", "all_epochs", "D.pt"), map_location=torch.device('cpu'))
    G_optimizer = torch.load(os.path.join(args.continue_with, "params", "all_epochs", "G_optimizer.pt"), map_location=torch.device('cpu'))
    D_optimizer = torch.load(os.path.join(args.continue_with, "params", "all_epochs", "D_optimizer.pt"), map_location=torch.device('cpu'))

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
    ModelSaver(output_path, n=10, overwrite=False, print_output=True),
    KillSwitchListener(output_path)
]
train_loop = VEEGANTrainLoop(
    listeners=listeners,
    Gz=Gz,
    Gx=Gx,
    D=D,
    optim_G=G_optimizer,
    optim_D=D_optimizer,
    dataloader=dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    d_img_noise_std=args.instance_noise_std,
    pre_training_steps=args.pre_train_steps

)

train_loop.train()
