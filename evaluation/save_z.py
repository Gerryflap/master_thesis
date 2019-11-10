"""
    This is a simple script that can save a z.npy which can be loaded into the latent space explorer
"""
import argparse

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image

import dlib

parser = argparse.ArgumentParser(description="Image to latent vector converter.")
parser.add_argument("--enc", action="store", type=str, help="Path to Gz/Encoder model")
parser.add_argument("--img", action="store", type=str, help="Path to input image")
parser.add_argument("--res", action="store", type=int, help="Model input resolution")
parser.add_argument("--sample", action="store_true", default=False,
                    help="If true, a sample is taken from the encoder output distribution. Otherwise the mean is used.")
parser.add_argument("--fix_contrast", action="store_true", default=False,
                    help="If true, makes sure that the colors in the image span from 0-255")
args = parser.parse_args()

fname_enc = args.enc
fname_img = args.img

# filename_enc = "results/celeba64/ali/2019-11-08T14:59:59/params/all_epochs/Gz.pt"

Gz = torch.load(fname_enc, map_location=torch.device('cpu'))

predictor_path = "data/data_prep/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

z_size = Gz.latent_size
resolution = args.res
real_resolution = resolution

frame = Image.open(fname_img)
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

crop_region_size = 10
frame = dlib.get_face_chip(frame, faces[0], size=(64 + crop_region_size*2))

frame = frame[crop_region_size:-crop_region_size, crop_region_size:-crop_region_size]

if args.fix_contrast:
    frame = frame.astype(np.float32)
    imin, imax = np.min(frame), np.max(frame)
    frame -= imin
    frame *= 255.0/(imax - imin)
    frame = frame.astype(np.uint8)


frame = Image.fromarray(frame)
input_frame = tvF.scale(frame, real_resolution)
input_frame = tvF.to_tensor(input_frame).float()

# input_frame = input_frame.permute(2, 0, 1)
input_frame = input_frame.unsqueeze(0)
# input_frame /= 255.0
input_frame = input_frame * 2 - 1
z, z_mean, z_logvar = Gz(input_frame)
if not args.sample:
    z = z_mean

z = z.detach().numpy()
np.save("z.npy", z)