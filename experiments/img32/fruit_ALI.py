from data.FruitDataset import FruitDataset
from trainloops.ali_train_loop import ALITrainLoop
from data.celeba_cropped import CelebaCropped
import util.output
from torchvision import transforms
import torch
import argparse
import models.conv32.models as models

# Parse commandline arguments
from trainloops.listeners.ae_image_sample_logger import AEImageSampleLogger
from trainloops.listeners.loss_reporter import LossReporter
from trainloops.listeners.model_saver import ModelSaver

parser = argparse.ArgumentParser(description="Celeba ALI experiment.")
parser.add_argument("--batch_size", action="store", type=int, default=100, help="Changes the batch size, default is 100")
parser.add_argument("--lr", action="store", type=float, default=0.0001,
                    help="Changes the learning rate, default is 0.0001")
parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
parser.add_argument("--l_size", action="store", type=int, default=256, help="Size of the latent space")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="Enables CUDA support. The script will fail if cuda is not available")
parser.add_argument("--dropout_rate", action="store", default=0.2, type=float,
                    help="Sets the dropout rate in D")

args = parser.parse_args()

output_path = util.output.init_experiment_output_dir("fruit32", "ali", args)

dataset = FruitDataset("data/fruit/32x32/", transform=transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

Gz = models.Encoder(args.l_size, True)
Gx = models.Generator(args.l_size)
D = models.Discriminator(args.l_size, args.dropout_rate, 1)
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
    AEImageSampleLogger(output_path, dataset, args, folder_name="AE_samples_train"),
    ModelSaver(output_path, n=1, overwrite=True, print_output=True)
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
    d_img_noise_std=0.1,
    decrease_noise=True,
    use_sigmoid=True
)

train_loop.train()
