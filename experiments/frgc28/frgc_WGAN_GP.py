from data.frgc_cropped import FRGCCropped
from models.conv28.discriminator import Discriminator28
from trainloops.wgangp_train_loop import GanTrainLoop
from models.conv28.ali_discriminator import ALIDiscriminator28
from models.conv28.generator import Generator28
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.gan_image_sample_logger import GanImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

# Parse commandline arguments


parser = argparse.ArgumentParser(description="FRGC WGANGP experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=1000, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=128, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--use_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D")
parser.add_argument("--dropout_rate", action="store", default=0.0, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--use_lr_norm", action="store_true", default=False,
                    help="Uses local response norm, which will make the generator samples "
                         "independent from the rest of the batch.")
parser.add_argument("--lambd", action="store", type=float, default=10.0,
                    help="Lambda, multiplier for gradient penalty")
parser.add_argument("--d_steps", action="store", type=int, default=2,
                    help="Amount of discriminator steps per generator step")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("frgc28", "WGAN_GP", args)

dataset = FRGCCropped( download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))


valid_dataset = FRGCCropped( download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Dataset length: ", len(dataset))

Gx = Generator28(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True, use_lr_norm=args.use_lr_norm)
D = Discriminator28(args.l_size, args.h_size, use_mish=args.use_mish, n_channels=3,
                       dropout=args.dropout_rate)
G_optimizer = torch.optim.Adam(Gx.parameters(), lr=args.lr, betas=(0.0, 0.9))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

if args.cuda:
    Gx = Gx.cuda()
    D = D.cuda()

Gx.init_weights()
D.init_weights()

listeners = [
    LossReporter(),
    GanImageSampleLogger(output_path, args, every_n_epochs=10),
    # DiscriminatorOverfitMonitor(dataset, valid_dataset, 100, args),
    ModelSaver(output_path, n=20, overwrite=True, print_output=True),
]


train_loop = GanTrainLoop(
    listeners=listeners,
    G=Gx,
    D=D,
    G_optimizer=G_optimizer,
    D_optimizer=D_optimizer,
    dataloader=dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    D_steps_per_G_step=args.d_steps,
    lambd=args.lambd

)

train_loop.train()
