"""

    WARNING: not yet functional because the MNIST dataset does give the correct items yet.
    Morphing GAN on the MNIST dataset.
"""
from data.mnist_pairs import MNISTPairs
from models.conv28.encoder import Encoder28

from trainloops.listeners.morph_image_logger import MorphImageLogger
from trainloops.morphing_gan_train_loop import MorphingGANTrainLoop
from models.conv28.ali_discriminator import ALIDiscriminator28
from models.conv28.generator import Generator28
import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="MNIST Morphing GAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--fc_h_size", action="store", type=int, default=None,
                    help="Sets the fc_h_size, which changes the size of the fully connected layers in D")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
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
parser.add_argument("--morgan_alpha", action="store", default=0.0, type=float,
                    help="Sets the alpha parameter in the MorGAN training algorithm")
parser.add_argument("--morph_loss_factor", action="store", default=0.3, type=float,
                    help="Scales the morph loss")
parser.add_argument("--use_dis_l_reconstruction_loss", action="store_true", default=False,
                    help="Switches the reconstruction loss to a VAEGAN like loss instead of pixelwise.")
parser.add_argument("--use_dis_l_morph_loss", action="store_true", default=False,
                    help="Switches the morph loss to a VAEGAN like loss instead of pixelwise.")
args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("mnist", "Morphing_GAN", args)

dataset = MNISTPairs("data/downloads/mnist", train=True, download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))

valid_dataset = MNISTPairs("data/downloads/mnist", train=False, download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

Gz = Encoder28(args.l_size, args.h_size, args.use_mish, n_channels=1)
Gx = Generator28(args.l_size, args.h_size, args.use_mish, False, n_channels=1, sigmoid_out=True)
D = ALIDiscriminator28(args.l_size, args.h_size, use_bn=args.use_batchnorm_in_D, use_mish=args.use_mish, n_channels=1, dropout=args.dropout_rate, fc_h_size=args.fc_h_size)
G_optimizer = torch.optim.Adam(list(Gz.parameters()) + list(Gx.parameters()), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

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
    MorphImageLogger(output_path, valid_dataset, args),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True)
]
train_loop = MorphingGANTrainLoop(
    listeners=listeners,
    Gz=Gz,
    Gx=Gx,
    D=D,
    optim_G=G_optimizer,
    optim_D=D_optimizer,
    dataloader=dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    morgan_alpha=args.morgan_alpha,
    d_img_noise_std=0.1,
    use_sigmoid=True,
    morph_loss_factor=args.morph_loss_factor,
    reconstruction_loss_mode="pixelwise" if not args.use_dis_l_reconstruction_loss else "dis_l",
    morph_loss_mode="pixelwise" if not args.use_dis_l_morph_loss else "dis_l"

)

train_loop.train()
