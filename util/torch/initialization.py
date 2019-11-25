import torch.nn as nn


def weights_init(m):
    # This was taken from the PyTorch DCGAN tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # The value for stddev has been altered to be equal to the ALI value
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight.data, 1.0, 0.01)
            nn.init.constant_(m.bias.data, 0)