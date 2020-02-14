from data.frgc_cropped import FRGCCropped
from models.conv28.encoder import Encoder28
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


parser = argparse.ArgumentParser(description="FRGC MorGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=32,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--fc_h_size", action="store", type=int, default=None,
                    help="Sets the fc_h_size, which changes the size of the fully connected layers in D")
parser.add_argument("--epochs", action="store", type=int, default=1000, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=128, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--use_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D")
parser.add_argument("--dropout_rate", action="store", default=0.2, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--morgan_alpha", action="store", default=0.3, type=float,
                    help="Sets the alpha_z parameter of MorGAN")
parser.add_argument("--instance_noise_std", action="store", default=0.0, type=float,
                    help="Sets the standard deviation for instance noise (noise added to inputs of D)")
parser.add_argument("--d_real_label", action="store", default=1.0, type=float,
                    help="Changes the label value for the \"real\" output of D. "
                         "This can be used for label smoothing. "
                         "Recommended is 1.0 for no smoothing or 0.9 for smoothing")
parser.add_argument("--use_dis_l_reconstruction_loss", action="store_true", default=False,
                    help="Switches the reconstruction loss to a VAEGAN like loss instead of pixelwise.")
parser.add_argument("--frs_path", action="store", default=None, help="Path to facial recognition system model. "
                                                                     "Switches to FRS reconstruction loss")
parser.add_argument("--use_lr_norm", action="store_true", default=False,
                    help="Uses local response norm, which will make the generator and encoder samples "
                         "independent from the rest of the batch.")
parser.add_argument("--r1_gamma", action="store", default=0.0, type=float,
                    help="If > 0, enables R1 loss which pushes the gradient "
                         "norm to zero for real samples in the discriminator.")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("frgc28", "MorGAN", args)

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

Gz = Encoder28(args.l_size, args.h_size, args.use_mish, n_channels=3, cap_variance=True, use_lr_norm=args.use_lr_norm)
Gx = Generator28(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True, use_lr_norm=args.use_lr_norm)
D = ALIDiscriminator28(args.l_size, args.h_size, use_bn=args.use_batchnorm_in_D, use_mish=args.use_mish, n_channels=3,
                       dropout=args.dropout_rate, fc_h_size=args.fc_h_size)
G_optimizer = torch.optim.Adam(list(Gz.parameters()) + list(Gx.parameters()), lr=args.lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Code for loading frs model when frs based reconstruction loss is used
frs_model = None
if args.frs_path is not None:
    frs_model = torch.load(args.frs_path)
    frs_model.eval()
    if args.cuda:
        frs_model = frs_model.cuda()

if args.cuda:
    Gz = Gz.cuda()
    Gx = Gx.cuda()
    D = D.cuda()

Gz.init_weights()
Gx.init_weights()
D.init_weights()

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True, every_n_epochs=10),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train", every_n_epochs=10),
    # DiscriminatorOverfitMonitor(dataset, valid_dataset, 100, args),
    ModelSaver(output_path, n=10, overwrite=True, print_output=True),
]

reconstruction_loss_mode = "pixelwise" if not args.use_dis_l_reconstruction_loss else "dis_l"
if frs_model is not None:
    reconstruction_loss_mode = "frs"

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
    use_sigmoid=True,
    reconstruction_loss_mode=reconstruction_loss_mode,
    frs_model=frs_model,
    r1_reg_gamma=args.r1_gamma
)

train_loop.train()
