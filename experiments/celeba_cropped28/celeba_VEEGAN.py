from models.conv28.encoder import Encoder28
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.veegan_train_loop import VEEGANTrainLoop
from models.conv28.veegan_discriminator import VEEGANDiscriminator28
from models.conv28.generator import Generator28
from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse

# Parse commandline arguments
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="Celeba VEEGAN experiment.")
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
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--use_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D")
parser.add_argument("--dropout_rate", action="store", default=0.2, type=float,
                    help="Sets the dropout rate on the input of the first fully connected layer of D")
parser.add_argument("--instance_noise_std", action="store", default=0.0, type=float,
                    help="Sets the standard deviation for instance noise (noise added to inputs of D)")
parser.add_argument("--pre_train_steps", action="store", type=int, default=0, help="Number of pre training steps for Gz")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba28", "VEEGAN", args)

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

Gz = Encoder28(args.l_size, args.h_size, args.use_mish, n_channels=3)
Gx = Generator28(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True)
D = VEEGANDiscriminator28(args.l_size, args.h_size, use_bn=args.use_batchnorm_in_D, use_mish=args.use_mish, n_channels=3, dropout=args.dropout_rate, fc_h_size=args.fc_h_size)
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
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    ModelSaver(output_path, n=20, overwrite=False, print_output=True),
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
