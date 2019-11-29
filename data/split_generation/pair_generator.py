"""
    This script is intended to be run with an OpenFace model as input
"""
import argparse
import openface
import numpy as np

parser = argparse.ArgumentParser(description="Image pair selector")
parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")

args = parser.parse_args()

align = openface.AlignDlib("data/data_prep")
net = openface.TorchNeuralNet(args.networkModel, args.imgDim, cuda=args.cuda)

# `img` is a numpy matrix containing the RGB pixels of the image.
bb = align.getLargestFaceBoundingBox(img)
alignedFace = align.align(args.imgDim, img, bb,
                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
rep1 = net.forward(alignedFace)

# `rep2` obtained similarly.
d = rep1 - rep2
distance = np.dot(d, d)