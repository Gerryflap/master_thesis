# This file is based on a tutorial for dlib (http://dlib.net/face_alignment.py.html), but is heavily modified.
import os

from PIL import Image


def crop_images(
        input_dir,
        output_dir
):
    import dlib
    predictor_path = "data/data_prep/shape_predictor_5_face_landmarks.dat"
    # Crop size. The full face image will have the resolution 64 + 2*crop_size and then crop_size pixels will be cut off at the top, bottom, left and right
    # Is used to zoom in on facial features
    crop_size = 0

    # Load all the models we need: a detector to find the faces, a shape predictor
    # to find face landmarks so we can precisely localize the face
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    filelist = list(os.listdir(input_dir))
    print(len(filelist))
    skipped = 0

    for i, fname in enumerate(filelist):
        if fname[-4:] != ".png" and fname[-4:] != ".jpg":
            skipped += 1
            continue

        if os.path.isfile(output_dir + fname):
            continue

        # Load the image using Dlib
        img = dlib.load_rgb_image(input_dir + fname)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)

        num_faces = len(dets)
        if num_faces == 0:
            skipped += 1
            continue

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))

        image = dlib.get_face_chip(img, faces[0], size=64 + 2 * crop_size)
        if crop_size != 0:
            image = image[crop_size:-crop_size, crop_size:-crop_size]
        img_obj = Image.fromarray(image)
        img_obj.save(output_dir + fname)

        if i % 1000 == 0:
            print("%d/%d\t\tskipped: %d" % (i, len(filelist), skipped))
