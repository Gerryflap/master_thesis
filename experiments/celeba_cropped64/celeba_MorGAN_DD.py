from models.conv28.encoder import Encoder28
from models.conv64.discriminator import Discriminator64
from models.conv64_ali.encoder import Encoder64
from models.conv64_ali.generator import Generator64
from models.mixture.discriminator import Discriminator
from trainloops.MorGAN_dual_discriminator_train_loop import MorGANDDTrainLoop
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.wgangp_train_loop import GanTrainLoop
from models.conv28.discriminator import Discriminator28
from models.conv28.generator import Generator28
from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from trainloops.listeners.gan_image_sample_logger import GanImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="Celeba MorGAN_DD experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=64,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--d_steps", action="store", type=int, default=1,
                    help="D steps per G step")
parser.add_argument("--epochs", action="store", type=int, default=123, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=512, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--alpha_x", action="store", type=float, default=0.3,
                    help="x reconstruction loss scaling factor")
parser.add_argument("--alpha_z", action="store", type=float, default=0.3,
                    help="z reconstruction loss scaling factor")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--dropout_rate", action="store", default=0.0, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--use_dis_l_reconstruction_loss", action="store_true", default=False,
                    help="Switches the x reconstruction loss to a VAEGAN like loss instead of pixelwise.")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "MorGAN_DD", args)

dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", morgan_like_filtering=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

Gx = Generator64(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True)
Gz = Encoder64(args.l_size, args.h_size, args.use_mish, n_channels=3)

Dx = Discriminator64(args.h_size, use_bn=True, use_mish=args.use_mish, n_channels=3, dropout=args.dropout_rate, use_logits=True)
Dz = Discriminator(args.l_size, h_size=256, input_size=args.l_size, batchnorm=True)

G_optimizer = torch.optim.Adam(list(Gx.parameters()) + list(Gz.parameters()), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(list(Dx.parameters()) + list(Dz.parameters()), lr=args.lr, betas=(0.5, 0.999))

if args.cuda:
    Gx = Gx.cuda()
    Dx = Dx.cuda()
    Gz = Gz.cuda()
    Dz = Dz.cuda()

Gx.init_weights()
Gz.init_weights()
Dx.init_weights()
Dz.init_weights()

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    ModelSaver(output_path, n=5, overwrite=True, print_output=True),
    KillSwitchListener(output_path)
]

trainloop = MorGANDDTrainLoop(
    listeners,
    Gz,
    Gx,
    Dz,
    Dx,
    G_optimizer,
    D_optimizer,
    dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    D_steps_per_G_step=args.d_steps,
    alpha_x=args.alpha_x,
    alpha_z=args.alpha_z,
    use_dis_l_x_reconstruction_loss=args.use_dis_l_reconstruction_loss
)

trainloop.train()
