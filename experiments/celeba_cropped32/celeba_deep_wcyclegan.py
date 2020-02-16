from models.mixture.discriminator import Discriminator
from models.stylegan2.stylegan2_like_discriminator import DeepDiscriminator
from models.stylegan2.stylegan2_like_encoder import DeepEncoder
from models.stylegan2.stylegan2_like_generator import DeepGenerator
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

parser = argparse.ArgumentParser(description="Celeba Deep wcyclegan experiment.")
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
parser.add_argument("--alpha_x", action="store", default=0.3, type=float,
                    help="Sets the alpha (reconstruction) parameter for x")
parser.add_argument("--alpha_z", action="store", default=0.0, type=float,
                    help="Sets the alpha (reconstruction) parameter for z")
parser.add_argument("--d_steps", action="store", type=int, default=5, help="Number of D steps per G step")


args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("celeba32", "deep_wcyclegan", args)

dataset = CelebaCropped(split="train", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
]))

valid_dataset = CelebaCropped(split="valid", download=True, morgan_like_filtering=True, transform=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

print("Dataset length: ", len(dataset))

Gz = DeepEncoder(args.l_size, args.h_size, 32, 3, lrn=True)
Gx = DeepGenerator(args.l_size, args.h_size, 4, 3, lrn=True)
Dx = DeepDiscriminator(args.h_size, 32, 3, bn=False)
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
Dz.init_weights()
Dx.init_weights()

listeners = [
    LossReporter(),
    AEImageSampleLogger(output_path, valid_dataset, args, folder_name="AE_samples_valid", print_stats=True),
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    # DiscriminatorOverfitMonitor(dataset, valid_dataset, 100, args),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True),
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
    alpha_x=args.alpha_x,
    alpha_z=args.alpha_z,
    D_steps_per_G_step=args.d_steps
)

train_loop.train()
