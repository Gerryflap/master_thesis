import torch


class LatentVector(torch.nn.Module):
    def __init__(self, batch_size, l_size):
        super().__init__()
        self.param = torch.nn.Parameter(torch.normal(0, 1, (batch_size, l_size)), requires_grad=True)

    def forward(self, inp):
        return self.param


def optimize_z_batch(Gx, x1, x2=None, n_steps=1000, lr=0.01, starting_z=None, dis_l_D=None, cuda=False):
    """
    Optimizes a batch of z values to resemble x1 as close as possible or to morph x1 and x2 in the best way using Gx.
    By default pixel-wise l1-loss is used.
    :param Gx: The generator Gx
    :param x1: A batch of images
    :param x2: Another batch of images. When given, this method will optimize the morph loss instead of reconstruction loss
    :param n_steps: Number of optimization steps
    :param lr: Learning rate for Adam
    :param starting_z: Overrides the random normal initialization values for z. Can be used to start with the output of Gz.
    :param dis_l_D: A Discriminator that has a compute_dx method. When given, dis_l loss is used instead of pixelwise l1
    :return: A batch of optimized z values and an info object for loss reporting
        (loss object is empty at the moment, might be implemented later.)
    """

    z = LatentVector(x1.size(0), Gx.latent_size)

    Gx.requires_grad = False
    if cuda:
        z = z.cuda()

    if starting_z is not None:
        z.param.data = starting_z.detach()

    if dis_l_D is not None:
        _, x1_dis_l = dis_l_D.compute_dx(x1)
        x1_dis_l = x1_dis_l.detach()
        if x2 is not None:
            _, x2_dis_l = dis_l_D.compute_dx(x2)
            x2_dis_l = x2_dis_l.detach()

    opt = torch.optim.Adam(z.parameters(), lr)

    for step in range(n_steps):
        z_val = z.forward(None)
        x_out = Gx(z_val)

        if dis_l_D is None:
            loss = torch.nn.functional.l1_loss(x_out, x1)

            if x2 is not None:
                loss += torch.nn.functional.l1_loss(x_out, x2)
                loss *= 0.5
        else:
            _, x_out_dis_l = dis_l_D.compute_dx(x_out)
            loss = torch.nn.functional.mse_loss(x_out_dis_l, x1_dis_l)

            if x2 is not None:
                loss += torch.nn.functional.mse_loss(x_out_dis_l, x2_dis_l)
                loss *= 0.5
        opt.zero_grad()
        loss.backward()
        opt.step()
    return z.forward(None).detach(), None
