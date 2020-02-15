from models.stylegan2.stylegan2_like_ali_discriminator import DeepAliDiscriminator
from models.stylegan2.stylegan2_like_encoder import DeepEncoder
from models.stylegan2.stylegan2_like_generator import DeepGenerator
from trainloops.ali_train_loop import ALITrainLoop
from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.cluster_killswitch import KillSwitchListener
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

# Parse commandline arguments


parser = argparse.ArgumentParser(description="Celeba Deep MorGAN experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=65, help="Changes the batch size, default is 65")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--h_size", action="store", type=int, default=64,
                    help="Sets the h_size, which changes the size of the network")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=256, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--use_mish", action="store_true", default=False,
                    help="Changes all activations except the ouput of D and G to mish, which might work better")
parser.add_argument("--disable_batchnorm_in_D", action="store_true", default=False,
                    help="Disables batch normalization in D")
parser.add_argument("--morgan_alpha", action="store", default=0.3, type=float,
                    help="Sets the alpha parameter of MorGAN")
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
parser.add_argument("--r1_gamma", action="store", default=0.0, type=float,
                    help="If > 0, enables R1 loss which pushes the gradient "
                         "norm to zero for real samples in the discriminator.")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba32", "Deep_MorGAN", args)

dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Dataset length: ", len(dataset))

Gz = DeepEncoder(args.l_size, args.h_size, 64, 4, bn=True)
Gx = DeepGenerator(args.l_size, args.h_size, 4, 4, bn=True)
D = DeepAliDiscriminator(args.l_size, args.h_size, 64, 4, bn=not args.disable_batchnorm_in_D)
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
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    # DiscriminatorOverfitMonitor(dataset, valid_dataset, 100, args),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
    KillSwitchListener(output_path)
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
