"""
    This is a simple script that uses gradient descend on a latent vector to match the image.
     It saves z.npy which can be loaded into the latent space explorer
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image

import dlib
from torchvision.transforms import transforms

from util.torch.losses import euclidean_distance

parser = argparse.ArgumentParser(description="Image to latent vector converter.")

parser.add_argument("--param_path", action="store", type=str, required=True, help="Path to Gx and D models")
parser.add_argument("--img", action="store", type=str, required=True, help="Path to input image")
parser.add_argument("--img2", action="store", type=str, default=None, help="Path to second input image for morphing")
parser.add_argument("--res", action="store", type=int, required=True, help="Model input resolution")
parser.add_argument("--lr", action="store", type=float, default=0.05, help="Adam learning rate")
parser.add_argument("--n_steps", action="store", type=int, default=1000, help="Number of optimization steps to take")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="If true, a sample is taken from the encoder output distribution. Otherwise the mean is used.")
parser.add_argument("--fix_contrast", action="store_true", default=False,
                    help="If true, makes sure that the colors in the image span from 0-255")
parser.add_argument("--sigmoid_model", action="store_true", default=False,
                    help="This flag is required if the loaded model was trained on images with range 0-1. This is done with the MorGAN models at the time of writing.")
parser.add_argument("--visualize", action="store_true", default=False,
                    help="Visualizes the images every step")
parser.add_argument("--use_dis_l", action="store_true", default=False,
                    help="Switches to Dis_l loss instead of pixels")
parser.add_argument("--regularization_factor", action="store", type=float, default=0.03,
                    help="Scales the regularization term "
                         "that keeps z close to 0")
parser.add_argument("--init_with_Gz", action="store_true", default=False,
                    help="Initializes z on Gz(x) or 0.5*(Gz(x1) + Gz(x2))")
parser.add_argument("--frs_path", action="store", default=None, help="Path to facial recognition system model. "
                                                                     "Switches to FRS reconstruction loss")
parser.add_argument("--d_real_regularization", action="store_true", default=False, help="Keeps D(x,z) to zero")
parser.add_argument("--no_face", action="store_true", default=False, help="Skips facial recognition and just rescales the image.")
parser.add_argument("--show_loss_plots", action="store_true", default=False, help="When enabled, shows a plot off different losses over the course of optimizing.")
parser.add_argument("--use_frs_l", action="store_true", default=False, help="Use dis_l like loss with the FRS. This will disable the FRS embedding loss")

args = parser.parse_args()

fname_dec = os.path.join(args.param_path, "Gx.pt")
if not os.path.exists(fname_dec):
    fname_dec = os.path.join(args.param_path, "G.pt")
if not os.path.exists(fname_dec):
    fname_dec = os.path.join(args.param_path, "dec.pt")

Gx = torch.load(fname_dec, map_location=torch.device('cpu'))
Gx.eval()
Gx.requires_grad = False

D = None

# Code for loading frs model when frs based reconstruction loss is used
frs_model = None
if args.frs_path is not None:
    frs_model = torch.load(args.frs_path)
    frs_model.requires_grad = False
    frs_model.eval()
    if args.cuda:
        frs_model = frs_model.cuda()

if args.use_dis_l or args.d_real_regularization:
    fname_D = os.path.join(args.param_path, "D.pt")

    D = torch.load(fname_D, map_location=torch.device('cpu'))
    D.eval()

Gz = None
if args.init_with_Gz:
    fname_Gz = os.path.join(args.param_path, "Gz.pt")

    Gz = torch.load(fname_Gz, map_location=torch.device('cpu'))
    Gz.eval()

if args.cuda:
    Gx = Gx.cuda()

    if D is not None:
        D = D.cuda()

    if args.init_with_Gz:
        Gz = Gz.cuda()

predictor_path = "data/data_prep/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

z_size = Gx.latent_size
resolution = args.res
real_resolution = resolution


def load_process_img(fname):
    frame = Image.open(fname)
    frame = np.array(frame)

    dets = detector(frame, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("No faces found!")
        exit()

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(frame, detection))

    crop_region_size = 0
    frame = dlib.get_face_chip(frame, faces[0], size=(64 + crop_region_size * 2)*5)

    if crop_region_size != 0:
        frame = frame[crop_region_size*5:-crop_region_size*5, crop_region_size*5:-crop_region_size*5]

    if args.fix_contrast:
        frame = frame.astype(np.float32)
        imin, imax = np.min(frame), np.max(frame)
        frame -= imin
        frame *= 255.0 / (imax - imin)
        frame = frame.astype(np.uint8)

    frame = Image.fromarray(frame)
    input_frame = tvF.resize(frame, real_resolution, interpolation=Image.LANCZOS)
    input_frame = tvF.to_tensor(input_frame).float()

    # input_frame = input_frame.permute(2, 0, 1)
    input_frame = input_frame.unsqueeze(0)
    # input_frame /= 255.0
    if not args.sigmoid_model:
        input_frame = input_frame * 2 - 1
    return input_frame


def load_process_img_no_face(fname):
    img = Image.open(fname)
    img = transforms.Resize(resolution, interpolation=Image.LANCZOS)(img)
    img = transforms.ToTensor()(img)
    return img


def numpy_euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


def to_numpy_img(img, tanh_mode):
    if tanh_mode:
        img = (img + 1) / 2
    img = np.moveaxis(img.detach().numpy(), 0, 2)
    img *= 255
    img = img.astype(np.uint8)
    return img


class LatentVector(torch.nn.Module):
    def __init__(self, l_size):
        super().__init__()
        self.param = torch.nn.Parameter(torch.normal(0, 1, (1, l_size)), requires_grad=True)

    def forward(self, inp):
        return self.param


stop = False


def stop_running(evt):
    global stop
    stop = True


if args.visualize:
    plt.figure().canvas.mpl_connect('close_event', stop_running)
    plt.ion()

x = load_process_img(args.img) if not args.no_face else load_process_img_no_face(args.img)

x2 = None
if args.img2 is not None:
    x2 = load_process_img(args.img2) if not args.no_face else load_process_img_no_face(args.img2)

if args.cuda:
    x = x.cuda()
    if x2 is not None:
        x2 = x2.cuda()

if args.use_dis_l:
    _, dis_l_x = D.compute_dx(x)
    D.requires_grad = False
    dis_l_x = dis_l_x.detach()

    if x2 is not None:
        _, dis_l_x2 = D.compute_dx(x2)
        dis_l_x2 = dis_l_x2.detach()

if frs_model is not None:
    emb_x = frs_model(x).detach()
    if x2 is not None:
        emb_x2 = frs_model(x2).detach()

if args.use_frs_l:
    frs_layer = 3
    frs_l_x = frs_model.get_frs_l(x, l=frs_layer)
    frs_model.requires_grad = False
    frs_l_x = frs_l_x.detach()

    if x2 is not None:
        frs_l_x2 = frs_model.get_frs_l(x2, l=frs_layer)
        frs_l_x2 = frs_l_x2.detach()


z = LatentVector(z_size)

if args.cuda:
    z = z.cuda()

if Gz is not None:
    if x2 is not None:
        z_val = Gz.morph(x, x2).detach()
    else:
        z_val = Gz.encode(x).detach()
    z.param.data = z_val

opt = torch.optim.Adam(z.parameters(), args.lr)

if args.show_loss_plots:
    import face_recognition
    recon_losses = []
    frs_dists1 = []
    img_ident = face_recognition.face_encodings(to_numpy_img(x[0].cpu(), False))[0]
    if x2 is not None:
        img2_ident = face_recognition.face_encodings(to_numpy_img(x2[0].cpu(), False))[0]
        frs_dists2 = []
        recon2_losses = []

try:
    for i in range(args.n_steps):
        z_val = z.forward(None)
        x_recon = Gx(z_val)

        loss = 0
        if args.use_dis_l:
            _, dis_l_x_rec = D.compute_dx(x_recon)
            loss = torch.nn.functional.mse_loss(dis_l_x_rec, dis_l_x, reduction="mean")
            loss1 = loss
            if x2 is not None:
                loss2 = torch.nn.functional.mse_loss(dis_l_x_rec, dis_l_x2, reduction="mean")
                loss = loss1 + loss2
                loss *= 0.5

        if args.use_frs_l:
            frs_l_x_rec = frs_model.get_frs_l(x_recon, l=frs_layer)

            loss1 = torch.nn.functional.mse_loss(frs_l_x_rec, frs_l_x, reduction="mean")
            if x2 is not None:
                loss2 = torch.nn.functional.mse_loss(frs_l_x_rec, frs_l_x2, reduction="mean")
                loss += 0.5 * (loss1 + loss2)
            else:
                loss += loss1

        if frs_model is not None and not args.use_frs_l:
            noise = torch.normal(0, 0.01, x_recon.size())
            if args.cuda:
                noise = noise.cuda()
            x_recon_noise = x_recon + noise
            emb_rec = frs_model(x_recon_noise)
            if x2 is None:
                dist = euclidean_distance(emb_rec, emb_x.detach())
                loss += dist
                print("Embedding distance: ", dist.detach().item())
            else:
                loss1 = euclidean_distance(emb_rec, emb_x)
                loss2 = euclidean_distance(emb_rec, emb_x2)
                print("Embedding distances: ", loss1.detach().item(), loss2.detach().item())
                # loss += torch.max(loss1, loss2)
                loss += torch.pow(loss1 ,2) + torch.pow(loss2, 2)
        if D is not None and args.d_real_regularization:
            #D_loss = torch.pow(D((x_recon, z_val)), 2).mean()
            D_loss = torch.sigmoid(D((x_recon, z_val.detach()))).mean()
            loss += D_loss
            print("D_loss", D_loss.detach().item())
        if D is None and frs_model is None:
            loss = torch.nn.functional.l1_loss(x_recon, x)
            loss1 = loss
            if x2 is not None:
                loss2 = torch.nn.functional.l1_loss(x_recon, x2, reduction="mean")
                loss += loss2
                loss *= 0.5

        # L_reg = torch.pow(z_val, 2).mean()
        L_reg = torch.pow(z_val.mean() - 0, 2) + torch.pow(z_val.var() - 1, 2)
        loss += args.regularization_factor * L_reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 1 == 0:
            if args.visualize:
                plt.clf()
                plt.imshow(to_numpy_img(x_recon[0].cpu(), False))
                plt.pause(0.01)

        if args.show_loss_plots:
            idents = face_recognition.face_encodings(to_numpy_img(x_recon[0].cpu(), False))
            if len(idents) != 1:
                dist = 2.0
            else:
                dist = numpy_euclidean_distance(img_ident, idents[0])
            frs_dists1.append(dist)
            if x2 is not None:
                if len(idents) != 1:
                    dist2 = 2.0
                else:
                    dist2 = numpy_euclidean_distance(img2_ident, idents[0])
                frs_dists2.append(dist2)
                recon_losses.append(loss1.detach().item())
                recon2_losses.append(loss2.detach().item())
            else:
                recon_losses.append(loss.detach().item())




        if stop:
            break
except KeyboardInterrupt:
    plt.close()
    z_val = z.param.data.detach().cpu().numpy()
    np.save("z.npy", z_val)

z = z.param.data.detach().cpu().numpy()
np.save("z.npy", z)

if args.show_loss_plots:
    print("Showing plots")
    plt.ioff()
    plt.clf()
    plt.plot(frs_dists1, color="red", label="FRS euclidean distance to x1")
    plt.plot(recon_losses, color="green", label="Recon loss wrt. x1")
    if x2 is not None:
        plt.plot(frs_dists2, color="orange", label="FRS euclidean distance to x2")
        plt.plot(recon2_losses, color="blue", label="Recon loss wrt. x2")
    plt.legend()
    plt.show()

plt.close()
