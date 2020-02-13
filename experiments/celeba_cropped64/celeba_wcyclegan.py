from data.FruitDataset import FruitDataset
from models.conv28.discriminator import Discriminator28
from models.conv28.encoder import Encoder28
from models.conv64.discriminator import Discriminator64
from models.conv64_ali.encoder import Encoder64
from models.conv64_ali.generator import Generator64
from models.mixture.discriminator import Discriminator
from trainloops.ali_train_loop import ALITrainLoop
from trainloops.gan_train_loop import GanTrainLoop
from models.conv28.ali_discriminator import ALIDiscriminator28
from models.conv28.generator import Generator28
from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

# Parse commandline arguments
from trainloops.wcyclegan_train_loop import WCycleGanTrainLoop

parser = argparse.ArgumentParser(description="Celeba WcycleGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=64,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=500, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=256, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--dropout_rate", action="store", default=0.0, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--alpha", action="store", default=1.0, type=float,
                    help="Sets the alpha parameter that scales the reconstruction loss")
parser.add_argument("--d_steps", action="store", type=int, default=5, help="D steps per G step")
parser.add_argument("--lambdx", action="store", type=float, default=10.0,
                    help="Lambda, multiplier for gradient penalty on x")
parser.add_argument("--lambdz", action="store", type=float, default=0.1,
                    help="Lambda, multiplier for gradient penalty on z")



args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "WCycleGAN", args)

dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Dataset length: ", len(dataset))

Gz = Encoder64(args.l_size, args.h_size, args.use_mish, n_channels=3, cap_variance=True, deterministic=True)
Gx = Generator64(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True)
Dx = Discriminator64(args.h_size, use_mish=args.use_mish, n_channels=3, dropout=args.dropout_rate, use_bn=False, use_logits=True)
Dz = Discriminator(args.l_size, 128, batchnorm=False, input_size=args.l_size)

G_optimizer = torch.optim.Adam(list(Gz.parameters()) + list(Gx.parameters()), lr=args.lr, betas=(0.0, 0.9))
D_optimizer = torch.optim.Adam(list(Dz.parameters()) + list(Dx.parameters()), lr=args.lr, betas=(0.0, 0.9))

if args.cuda:
    Gz = Gz.cuda()
    Gx = Gx.cuda()
    Dx = Dx.cuda()
    Dz = Dz.cuda()

Gz.init_weights()
Gx.init_weights()

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    ModelSaver(output_path, n=30, overwrite=False, print_output=True),
]
train_loop = WCycleGanTrainLoop(
    listeners=listeners,
    Gz=Gz,
    Gx=Gx,
    Dz=Dz,
    Dx=Dx,
    G_optimizer=G_optimizer,
    D_optimizer=D_optimizer,
    dataloader=dataloader,
    cuda=args.cuda,
    epochs=args.epochs,
    alpha=args.alpha,
    lambd_x=args.lambdx,
    lambd_z=args.lambdz,
    D_steps_per_G_step=args.d_steps
)

train_loop.train()
