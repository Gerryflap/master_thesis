import torch

from models.morphing_encoder import MorphingEncoder
from util.torch.initialization import weights_init


class Discriminator(MorphingEncoder):
    def __init__(self, latent_size, h_size=64, mode="normal"):
        super().__init__()
        self.latent_size = latent_size
        self.h_size = h_size
        self.mode = mode

        if self.mode not in {"normal", "ali", "vaegan"}:
            raise ValueError("Expected the discriminator mode to be one of {\"normal\", \"ali\", \"vaegan\"}.")

        self.Dx = torch.nn.Sequential(
            torch.nn.Linear(2, h_size),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(h_size, h_size),
            torch.nn.BatchNorm1d(h_size),
            torch.nn.LeakyReLU(0.02),
        )

        d_out_input_size = h_size
        if mode == "ali":
            self.Dz = torch.nn.Sequential(
                torch.nn.Linear(latent_size, h_size),
                torch.nn.LeakyReLU(0.02),

                torch.nn.Linear(h_size, h_size),
                torch.nn.LeakyReLU(0.02),
            )
            d_out_input_size = h_size*2

        self.D_out = torch.nn.Sequential(
            torch.nn.Linear(d_out_input_size, h_size),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(h_size, h_size),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(h_size, h_size),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(h_size, 1),
        )

    def compute_dx(self, x):
        # This method is used by the Morphing GAN algorithm to acquire the value for dis_l
        return None, self.Dx(x)

    def forward(self, inp):
        if self.mode == "ali":
            x, z = inp
            dz = self.Dz(z)
        else:
            x = inp

        dx = self.Dx(x)

        if self.mode == "ali":
            dxz_in = torch.cat([dx, dz], dim=1)
            return self.D_out(dxz_in)
        elif self.mode == "vaegan":
            d_out = self.D_out(dx)

            # We'll dx is an inbetween activation, which can be used as Dis_l(x)
            return d_out, dx
        else:
            d_out = self.D_out(dx)
            return d_out

    def init_weights(self):
        self.apply(weights_init)