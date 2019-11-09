import numpy as np


def gen_aligned_faces():
    print("Filtering unaligned faces...")
    with open("data/celeba/list_landmarks_align_celeba.txt", "r") as f:
        lines = f.readlines()

    lines_split = [line.split() for line in lines[2:]]

    fnames = []
    pos = np.zeros((len(lines_split), 6))
    w, h = 178, 218
    for i, features in enumerate(lines_split):
        fnames.append(features[0])
        left_eye_x = float(features[1])/w
        left_eye_y = float(features[2])/h
        right_eye_x = float(features[3])/w
        right_eye_y = float(features[4])/h
        nose_x = float(features[5])/w
        nose_y = float(features[6])/h
        pos[i] = [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y]

    left_distances = ((pos[:, 0] - pos[:, 4]) ** 2 + (pos[:, 1] - pos[:, 5]) ** 2) ** 0.5
    right_distances = ((pos[:, 2] - pos[:, 4]) ** 2 + (pos[:, 3] - pos[:, 5]) ** 2) ** 0.5

    print(left_distances.mean())

    diffs = np.abs(left_distances - right_distances)
    print(diffs.mean())

    # This number has been chosen to match the database size in MorGAN as closely as possible
    aligned = diffs <= 0.0288924
    print("Done... \nResulting database length: ", np.sum(aligned))

    fnames_f = {fname for i, fname in enumerate(fnames) if aligned[i]}
    return fnames_f

if __name__ == "__main__":
    gen_aligned_faces()