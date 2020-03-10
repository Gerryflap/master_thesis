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


def optimize_z_batch_recons(Gx, x1, x2, n_steps=1000, lr=0.01, starting_zs=None, dis_l_D=None, cuda=False):
    """
    Optimizes a batch of z values to resemble x1 as close as possible or to resemble x1 and x2 as good as possible.
    :param Gx: The generator Gx
    :param x1: A batch of images
    :param x2: Second batch of images.
    :param lr: Learning rate for Adam
    :param starting_zs: Tuple that overrides the random normal initialization values for z1 and z2. Can be used to start with the output of Gz.
    :param dis_l_D: A Discriminator that has a compute_dx method. When given, dis_l loss is used instead of pixelwise l1
    :return: A batch of optimized z1, z2, z_morph values and an info object for loss reporting
        (loss object is empty at the moment, might be implemented later.)
    """

    z1 = LatentVector(x1.size(0), Gx.latent_size)
    z2 = LatentVector(x2.size(0), Gx.latent_size)

    Gx.requires_grad = False
    if cuda:
        z1 = z1.cuda()
        z2 = z2.cuda()

    if starting_zs is not None:
        z1.param.data = starting_zs[0].detach()
        z2.param.data = starting_zs[1].detach()

    if dis_l_D is not None:
        _, x1_dis_l = dis_l_D.compute_dx(x1)
        x1_dis_l = x1_dis_l.detach()
        if x2 is not None:
            _, x2_dis_l = dis_l_D.compute_dx(x2)
            x2_dis_l = x2_dis_l.detach()

    opt = torch.optim.Adam(list(z1.parameters()) + list(z2.parameters()), lr)

    for step in range(n_steps):
        z1_val = z1.forward(None)
        z2_val = z2.forward(None)

        x1_out = Gx(z1_val)
        x2_out = Gx(z2_val)

        if dis_l_D is None:
            loss = torch.nn.functional.l1_loss(x1_out, x1) + \
                torch.nn.functional.l1_loss(x2_out, x2)
            loss *= 0.5
        else:
            _, x1_out_dis_l = dis_l_D.compute_dx(x1_out)
            _, x2_out_dis_l = dis_l_D.compute_dx(x2_out)

            loss = torch.nn.functional.mse_loss(x1_out_dis_l, x1_dis_l) + \
                torch.nn.functional.mse_loss(x2_out_dis_l, x2_dis_l)
            loss *= 0.5
        opt.zero_grad()
        loss.backward()
        opt.step()
    z1_val = z1.forward(None).detach()
    z2_val = z2.forward(None).detach()
    z_morph = 0.5 * (z1_val + z2_val)
    return (z1_val, z2_val, z_morph), None
