"""
    This is a simple script that uses gradient descend on a latent vector to match the image.
     It saves z.npy which can be loaded into the latent space explorer
"""
import argparse
import os

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image

import dlib

from util.torch.losses import euclidean_distance

parser = argparse.ArgumentParser(description="Image to latent vector converter.")

parser.add_argument("--param_path", action="store", type=str, required=True, help="Path to D model folder")
parser.add_argument("--img", action="store", type=str, required=True, help="Path to input image")
parser.add_argument("--img2", action="store", type=str, default=None, help="Path to second input image for morphing")
parser.add_argument("--lr", action="store", type=float, default=0.05, help="Adam learning rate")
parser.add_argument("--n_steps", action="store", type=int, default=1000, help="Number of optimization steps to take")
parser.add_argument("--res", action="store", type=int, default=64, help="Image resolution")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="If true, a sample is taken from the encoder output distribution. Otherwise the mean is used.")
parser.add_argument("--fix_contrast", action="store_true", default=False,
                    help="If true, makes sure that the colors in the image span from 0-255")
parser.add_argument("--visualize", action="store_true", default=False,
                    help="Visualizes the images every step")
parser.add_argument("--use_dis_l", action="store_true", default=False,
                    help="Switches to Dis_l loss instead of pixels")
parser.add_argument("--frs_path", action="store", default=None, help="Path to facial recognition system model. "
                                                                     "Switches to FRS reconstruction loss")

args = parser.parse_args()


D = None

# Code for loading frs model when frs based reconstruction loss is used
frs_model = None
if args.frs_path is not None:
    frs_model = torch.load(args.frs_path)
    frs_model.requires_grad = False
    frs_model.eval()
    if args.cuda:
        frs_model = frs_model.cuda()

if args.use_dis_l:
    fname_D = os.path.join(args.param_path, "D.pt")

    D = torch.load(fname_D, map_location=torch.device('cpu'))
    D.eval()


if args.cuda:
    if D is not None:
        D = D.cuda()


predictor_path = "data/data_prep/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

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
    frame = dlib.get_face_chip(frame, faces[0], size=(64 + crop_region_size * 2))

    if crop_region_size != 0:
        frame = frame[crop_region_size:-crop_region_size, crop_region_size:-crop_region_size]

    if args.fix_contrast:
        frame = frame.astype(np.float32)
        imin, imax = np.min(frame), np.max(frame)
        frame -= imin
        frame *= 255.0 / (imax - imin)
        frame = frame.astype(np.uint8)

    frame = Image.fromarray(frame)
    input_frame = tvF.scale(frame, real_resolution)
    input_frame = tvF.to_tensor(input_frame).float()

    # input_frame = input_frame.permute(2, 0, 1)
    input_frame = input_frame.unsqueeze(0)
    # input_frame /= 255.0
    return input_frame


def to_numpy_img(img, tanh_mode):
    if tanh_mode:
        img = (img + 1) / 2
    img = np.moveaxis(img.detach().numpy(), 0, 2)
    img *= 255
    img = img.astype(np.uint8)
    return img


class TrainableImage(torch.nn.Module):
    def __init__(self, res):
        super().__init__()
        self.param = torch.nn.Parameter(torch.normal(0.0, 0.3, (1, 3, res, res)), requires_grad=True)
        self.param = self.param

    def forward(self, inp):
        return torch.sigmoid(self.param)


stop = False


def stop_running(evt):
    global stop
    stop = True


if args.visualize:
    import matplotlib.pyplot as plt

    plt.figure().canvas.mpl_connect('close_event', stop_running)
    plt.ion()

x = load_process_img(args.img)

x2 = None
if args.img2 is not None:
    x2 = load_process_img(args.img2)

if args.cuda:
    x = x.cuda()
    if x2 is not None:
        x2 = x2.cuda()

if D is not None:
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


x_rec = TrainableImage(resolution)
if args.cuda:
    x_rec = x_rec.cuda()

opt = torch.optim.Adam(x_rec.parameters(), args.lr)

for i in range(100):
    x_recon = x_rec.forward(None)
    loss = torch.nn.functional.mse_loss(x_recon, x)
    opt.zero_grad()
    loss.backward()
    opt.step()


try:
    for i in range(args.n_steps):
        x_recon = x_rec.forward(None)
        loss = 0
        if D is not None:
            _, dis_l_x_rec = D.compute_dx(x_recon)
            loss = torch.nn.functional.mse_loss(dis_l_x_rec, dis_l_x, reduction="mean")
            if x2 is not None:
                loss += torch.nn.functional.mse_loss(dis_l_x_rec, dis_l_x2, reduction="mean")
                loss *= 0.5

        if frs_model is not None:
            emb_rec = frs_model(x_recon)
            if x2 is None:
                loss += euclidean_distance(emb_rec, emb_x)
                print(loss.detach().item())
            else:
                loss1 = euclidean_distance(emb_rec, emb_x)
                loss2 = euclidean_distance(emb_rec, emb_x2)
                print(loss1.detach().item(), loss2.detach().item())
                loss += torch.max(loss1, loss2)
        if D is None and frs_model is None:
            loss = torch.nn.functional.mse_loss(x_recon, x)
            if x2 is not None:
                loss += torch.nn.functional.mse_loss(x_recon, x2, reduction="mean")
                loss *= 0.5

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 1 == 0:
            if args.visualize:
                plt.clf()
                plt.imshow(to_numpy_img(x_recon[0].cpu(), False))
                plt.pause(0.01)
        if stop:
            plt.close()
            break
except KeyboardInterrupt:
    plt.close()

