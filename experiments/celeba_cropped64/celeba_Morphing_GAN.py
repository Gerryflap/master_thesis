from data.celeba_cropped_pairs import CelebaCroppedPairs
from models.conv64_ali.encoder import Encoder64
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.morphing_gan_train_loop import MorphingGANTrainLoop
from models.conv64_ali.ali_discriminator import ALIDiscriminator64
from models.conv64_ali.generator import Generator64
from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

# Parse commandline arguments
parser = argparse.ArgumentParser(description="Celeba Morphing GAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=64,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--fc_h_size", action="store", type=int, default=None,
                    help="Sets the fc_h_size, which changes the size of the fully connected layers in D")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=256, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--disable_batchnorm_in_D", action="store_true", default=False,
                    help="Enables batch normalization in D")
parser.add_argument("--dropout_rate", action="store", default=0.2, type=float,
                    help="Sets the dropout rate in D")
parser.add_argument("--morgan_alpha", action="store", default=0.3, type=float,
                    help="Sets the alpha parameter of MorGAN")
parser.add_argument("--morph_loss_factor", action="store", default=0.3, type=float,
                    help="Scales the morph loss")
parser.add_argument("--use_dis_l_reconstruction_loss", action="store_true", default=False,
                    help="Switches the reconstruction loss to a VAEGAN like loss instead of pixelwise.")
parser.add_argument("--use_dis_l_morph_loss", action="store_true", default=False,
                    help="Switches the morph loss to a VAEGAN like loss instead of pixelwise.")
parser.add_argument("--instance_noise_std", action="store", default=0.1, type=float,
                    help="Sets the standard deviation for instance noise (noise added to inputs of D)")
parser.add_argument("--d_real_label", action="store", default=1.0, type=float,
                    help="Changes the label value for the \"real\" output of D. "
                         "This can be used for label smoothing. "
                         "Recommended is 1.0 for no smoothing or 0.9 for smoothing")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba64", "Morphing_GAN", args)

dataset = CelebaCroppedPairs(split="train", download=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

valid_dataset = CelebaCroppedPairs(split="valid", download=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Dataset length: ", len(dataset))

Gz = Encoder64(args.l_size, args.h_size, args.use_mish, n_channels=3, cap_variance=True)
Gx = Generator64(args.l_size, args.h_size, args.use_mish, n_channels=3, sigmoid_out=True)
D = ALIDiscriminator64(args.l_size, args.h_size, use_bn=not args.disable_batchnorm_in_D, use_mish=args.use_mish, n_channels=3, dropout=args.dropout_rate, fc_h_size=args.fc_h_size)
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
    ModelSaver(output_path, n=30, overwrite=False, print_output=True),
    KillSwitchListener(output_path)
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
    d_real_label=args.d_real_label,
    d_img_noise_std=args.instance_noise_std,
    decrease_noise=True,
    use_sigmoid=True,
    morph_loss_factor=args.morph_loss_factor,
    reconstruction_loss_mode="pixelwise" if not args.use_dis_l_reconstruction_loss else "dis_l",
    morph_loss_mode="pixelwise" if not args.use_dis_l_morph_loss else "dis_l"

)

train_loop.train()