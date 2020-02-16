from data.celeba_cropped_triplets import CelebaCroppedTriplets
from models.conv28.encoder import Encoder28
from models.conv28.ali_discriminator import ALIDiscriminator28
from models.conv28.generator import Generator28
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

# Parse commandline arguments
from trainloops.split_latent_morgan_train_loop import SplitMorGANTrainLoop

parser = argparse.ArgumentParser(description="Celeba split MorGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=16,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--fc_h_size", action="store", type=int, default=None,
                    help="Sets the fc_h_size, which changes the size of the fully connected layers in D")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=16, help="Size of the latent space")
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
parser.add_argument("--instance_noise_std", action="store", default=0.1, type=float,
                    help="Sets the standard deviation for instance noise (noise added to inputs of D)")
parser.add_argument("--d_real_label", action="store", default=1.0, type=float,
                    help="Changes the label value for the \"real\" output of D. "
                         "This can be used for label smoothing. "
                         "Recommended is 1.0 for no smoothing or 0.9 for smoothing")
parser.add_argument("--use_dis_l_reconstruction_loss", action="store_true", default=False,
                    help="Switches the reconstruction loss to a VAEGAN like loss instead of pixelwise.")
parser.add_argument("--l_constrained_size", action="store", type=int, default=12,
                    help="Sets the size of the constrained part of the latent space. "
                         "This should always be smaller or equal to the latent size.")
parser.add_argument("--l_constrained_factor", action="store", default=1.0, type=float,
                    help="The factor that scales L_latent (which is the loss that constrains a part of the latent space)")
parser.add_argument("--unconstrained_latent_noise_std", action="store", default=0.0, type=float,
                    help="Standard deviation of noise added to the unconstrained part of the latent space. "
                         "This is done to prevent the model from putting all information in that part")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba28", "Split_MorGAN", args)

dataset = CelebaCroppedTriplets(split="train", download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))

valid_dataset = CelebaCroppedTriplets(split="valid", download=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Dataset length: ", len(dataset))

Gz = Encoder28(args.l_size, args.h_size, args.use_mish, n_channels=3, cap_variance=True, add_dense_layer=True)
Gx = Generator28(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True, add_dense_layer=True)
D = ALIDiscriminator28(args.l_size, args.h_size, use_bn=args.use_batchnorm_in_D, use_mish=args.use_mish, n_channels=3, dropout=args.dropout_rate, fc_h_size=args.fc_h_size)
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
train_loop = SplitMorGANTrainLoop(
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
    reconstruction_loss_mode="pixelwise" if not args.use_dis_l_reconstruction_loss else "dis_l",
    constrained_latent_loss_factor=args.l_constrained_factor,
    constrained_latent_size=args.l_constrained_size,
    unconstrained_latent_noise_std=args.unconstrained_latent_noise_std

)

train_loop.train()
