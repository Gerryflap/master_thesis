import os
import face_recognition
import numpy as np


def run():
    assert os.path.exists("data")

    batch_size = 64

    with open("data/celeba_cropped/list_eval_partition_filtered.txt", "r") as f:
        lines = f.readlines()
        fnames = list([l.split()[0] for l in lines])

    print("Generating embeddings...")

    embeddings = []
    for i in range(0, len(fnames), batch_size):
        print(i)
        imax = min(i + batch_size, len(fnames))
        fname_batch = fnames[i:imax]
        images = list(
            [face_recognition.load_image_file("data/celeba/img_align_celeba/%s" % fname) for fname in fname_batch]
        )
        locations = face_recognition.batch_face_locations(images, batch_size=batch_size)

        batch_embeddings = []
        for j in range(len(fname_batch)):
            img = images[j]
            loc = locations[j]
            embedding_list = face_recognition.face_encodings(img, loc)
            if len(embedding_list) == 0:
                # Fill with nans
                embedding = np.full((128,), np.nan)
            else:
                embedding = embedding_list[0]
            batch_embeddings.append(embedding)
        embeddings.append(np.stack(batch_embeddings, axis=0))

    embeddings = np.concatenate(embeddings, axis=0)
    np.save("data/celeba_cropped/embeddings.npy", embeddings)
    with open("data/celeba_cropped/embedding_file_order.txt", "w") as f:
        f.write("\n".join(fnames))


if __name__ == "__main__":
    run()