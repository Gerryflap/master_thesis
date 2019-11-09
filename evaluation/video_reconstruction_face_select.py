import os

import cv2

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
import dlib


# 0 = No random, 1 = Every frame a new random
random_mode = 1
crop_region_size = 10

if random_mode == 0:
    rand_vec = 0
else:
    rand_vec = None


# image = None
orig_img = None
should_update = True

root = tk.Tk()
filename_enc = tk.filedialog.askopenfilename(initialdir="./results", title="Select encoder",
                                           filetypes=(("Pytorch model", "*.pt"), ("all files", "*.*")))
init_dir = os.path.split(filename_enc)[0]
filename_dec = tk.filedialog.askopenfilename(initialdir=init_dir, title="Select decoder",
                                           filetypes=(("Pytorch model", "*.pt"), ("all files", "*.*")))

root.destroy()

# filename_enc = "results/celeba64/ali/2019-11-08T14:59:59/params/all_epochs/Gz.pt"
# filename_dec = "results/celeba64/ali/2019-11-08T14:59:59/params/all_epochs/Gx.pt"

Gz = torch.load(filename_enc, map_location=torch.device('cpu'))
Gx = torch.load(filename_dec, map_location=torch.device('cpu'))

cap = cv2.VideoCapture(0)
predictor_path = "data/data_prep/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

z_size = Gx.latent_size
resolution = int(Gx(torch.zeros((z_size,))).size()[2])
print(resolution)
real_resolution = resolution


root = tk.Tk()
root.title("GAN webcam tool")
root.attributes('-type', 'dialog')

def update():
    global rand_vec
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = frame[:, ::-1, ::-1]

    dets = detector(frame, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("No faces found")
        root.after(1, update)
        return

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(frame, detection))

    frame = dlib.get_face_chip(frame, faces[0], size=(64 + crop_region_size*2))


    # frame = frame[::8, ::8]
    frame = frame[crop_region_size:-crop_region_size, crop_region_size:-crop_region_size]
    # input_frame = np.expand_dims(frame, axis=0)
    # input_frame = input_frame/255.0
    # # input_frame *= 5.0
    # # input_frame *= 2.8
    # input_frame *= 1.9
    # input_frame += 0.5 - np.mean(input_frame)
    # input_frame = np.clip(input_frame, 0, 1)

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
    if random_mode == 0:
        z = z_mean

    decoded = Gx(z)[0]
    decoded = (decoded + 1)/2
    decoded *= 255.0

    input_frame = (input_frame + 1)/2
    input_frame *= 255.0

    img = Image.fromarray((input_frame.permute(0, 2, 3, 1).detach().numpy()[0]).astype(np.uint8))
    img = img.resize((320, 320))
    img = ImageTk.PhotoImage(image=img)
    image = img

    img2 = Image.fromarray(decoded.detach().permute(1, 2, 0).numpy().astype(np.uint8))
    img2 = img2.resize((320, 320))
    img2 = ImageTk.PhotoImage(image=img2)
    image2 = img2

    root.image = image
    root.image2 = image2

    canvas.create_image(0, 0, anchor="nw", image=root.image)
    canvas2.create_image(0, 0, anchor="nw", image=root.image2)

    root.after(1, update)

frame = tk.Frame()
canvas = tk.Canvas(frame, width=320, height=320, bg='black')
canvas2 = tk.Canvas(frame, width=320, height=320, bg='black')
update()



frame.pack()
canvas.pack()
canvas2.pack()







try:
    tk.mainloop()

except KeyboardInterrupt:
    print("Done.")

# When everything done, release the capture
cap.release()