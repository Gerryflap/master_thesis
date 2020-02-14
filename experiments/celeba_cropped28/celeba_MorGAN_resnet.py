from torch.nn.modules import Flatten

from models.conv28.ali_discriminator import ALIDiscriminator28
from models.generic_models import Encoder, Generator, ALIDiscriminator
from trainloops.ali_train_loop import ALITrainLoop
from data.celeba_cropped import CelebaCropped
from util.residual_layer import ResidualConvolutionLayer, ResidualConvolutionTransposeLayer
import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.discriminator_overfit_monitor import DiscriminatorOverfitMonitor
from trainloops.listeners.gan_image_sample_logger import GanImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver
from trainloops.listeners.parameter_value_logger import ParameterValueLogger
from util.torch.activations import mish, Mish
from util.torch.modules import Reshape

parser = argparse.ArgumentParser(description="Celeba MorGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 65")
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
parser.add_argument("--dropout_rate", action="store", default=0.2, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--morgan_alpha", action="store", default=0.3, type=float,
                    help="Sets the alpha_z parameter of MorGAN")
parser.add_argument("--instance_noise_std", action="store", default=0.1, type=float,
                    help="Sets the standard deviation for instance noise (noise added to inputs of D)")
parser.add_argument("--d_real_label", action="store", default=1.0, type=float,
                    help="Changes the label value for the \"real\" output of D. "
                         "This can be used for label smoothing. "
                         "Recommended is 1.0 for no smoothing or 0.9 for smoothing")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba28", "MorGAN_resnet", args)

dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Dataset length: ", len(dataset))

h_size = args.h_size
latent_size = args.l_size

# Define the Gz network architecture
Gz_net = torch.nn.Sequential(
    torch.nn.Conv2d(3, h_size, 1),  # 28x28
    ResidualConvolutionLayer(h_size, h_size),
    ResidualConvolutionLayer(h_size, h_size),
    ResidualConvolutionLayer(h_size, h_size * 2, downscale=True),  # 14x14
    ResidualConvolutionLayer(h_size*2, h_size*2),
    ResidualConvolutionLayer(h_size*2, h_size*2),
    ResidualConvolutionLayer(h_size*2, h_size * 4, downscale=True),  # 7x7
    Flatten(),

    torch.nn.Linear(7*7*h_size*4, h_size*8),
    torch.nn.BatchNorm1d(h_size*8),
    Mish(),

    torch.nn.Linear(h_size*8, latent_size*2),
)

Gx_net = torch.nn.Sequential(
    torch.nn.Linear(latent_size, 7*7*h_size*4),
    Reshape(-1, h_size*4, 7, 7),

    ResidualConvolutionTransposeLayer(h_size * 4, h_size * 4),
    ResidualConvolutionTransposeLayer(h_size * 4, h_size * 4),
    ResidualConvolutionTransposeLayer(h_size * 4, h_size * 2, upscale=True),

    ResidualConvolutionTransposeLayer(h_size * 2, h_size * 2),
    ResidualConvolutionTransposeLayer(h_size * 2, h_size * 2),
    ResidualConvolutionTransposeLayer(h_size * 2, h_size, upscale=True),

    torch.nn.BatchNorm2d(h_size),
    Mish(),

    torch.nn.Conv2d(h_size, 3, 1),
    torch.nn.Sigmoid()
)

Dx_net = torch.nn.Sequential(
    torch.nn.Conv2d(3, h_size, 1),  # 28x28
    ResidualConvolutionLayer(h_size, h_size, bn=False),
    ResidualConvolutionLayer(h_size, h_size),
    ResidualConvolutionLayer(h_size, h_size * 2, downscale=True),  # 14x14
    ResidualConvolutionLayer(h_size * 2, h_size * 2),
    ResidualConvolutionLayer(h_size * 2, h_size * 2),
    ResidualConvolutionLayer(h_size * 2, h_size * 4, downscale=True),  # 7x7
    Flatten()
)

Dz_net = torch.nn.Sequential(
    torch.nn.Linear(latent_size, h_size * 16),
    Mish(),

    torch.nn.Linear(h_size * 16, h_size * 16),
    Mish(),
)

Dxz_net = torch.nn.Sequential(
    torch.nn.Linear(7*7*h_size*4 + h_size*16, h_size * 16),
    Mish(),

    torch.nn.Linear(h_size * 16, 1),
)

Gz = Encoder(Gz_net, args.l_size)
Gx = Generator(Gx_net, args.l_size)
D = ALIDiscriminator(args.l_size, Dx_net, Dz_net, Dxz_net)
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
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    # DiscriminatorOverfitMonitor(dataset, valid_dataset, 100, args),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
]
train_loop = ALITrainLoop(
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
    d_real_label=args.d_real_label,
    d_img_noise_std=args.instance_noise_std,
    decrease_noise=True,
    use_sigmoid=True
)

train_loop.train()
