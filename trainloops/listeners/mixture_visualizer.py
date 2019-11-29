import os

from trainloops.listeners.listener import Listener
import numpy as np
import torch
import matplotlib.pyplot as plt
import util


class MixtureVisualizer(Listener):
    def __init__(self, experiment_output_path, n_latent, valid_dataset, output_reproductions=False,
                 discriminator_output=False, d_output_resolution=100, cuda=False, sample_reconstructions=False,
                 every_n_epochs=10):
        super().__init__()
        self.path = os.path.join(experiment_output_path, "imgs", "mixture_outputs")
        util.output.make_result_dirs(self.path)
        self.valid_dataset = valid_dataset
        self.output_reproductions = output_reproductions
        self.discriminator_output = discriminator_output
        self.d_output_resolution = d_output_resolution
        self.sample_reconstructions = sample_reconstructions
        self.every_n_epochs = every_n_epochs
        self.cuda = cuda
        self.data = None
        self.static_z = torch.normal(0, 1, (len(valid_dataset), n_latent))
        if cuda:
            self.static_z = self.static_z.cuda()

    def initialize(self):
        # Assume that it is a mixture dataset, hence we can just grab the data directly
        self.data = self.valid_dataset.data

    def report(self, state_dict):
        if state_dict["epoch"]%self.every_n_epochs != 0:
            return
        plt.clf()
        if "G" in state_dict["networks"]:
            Gx = state_dict["networks"]["G"]
        elif "dec" in state_dict["networks"]:
            Gx = state_dict["networks"]["dec"]
        elif "Gx" in state_dict["networks"]:
            Gx = state_dict["networks"]["Gx"]
        else:
            raise ValueError("Could not find a decoder-like network in the state dict!")

        if self.output_reproductions:
            if "enc" in state_dict["networks"]:
                Gz = state_dict["networks"]["enc"]
            elif "Gz" in state_dict["networks"]:
                Gz = state_dict["networks"]["Gz"]
            else:
                raise ValueError("Could not find a encoder-like network in the state dict!")

        if self.discriminator_output:
            if "D" in state_dict["networks"]:
                D = state_dict["networks"]["D"]
            else:
                raise ValueError("Could not find a Discriminator in the state dict!")

            # Generate contourf
            xs = np.linspace(-1.2, 1.2, self.d_output_resolution, dtype=np.float32)
            ys = np.linspace(-1.2, 1.2, self.d_output_resolution, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            nn_inp = np.concatenate([xx.reshape((-1, 1)), yy.reshape((-1, 1))], axis=1)
            nn_inp = torch.from_numpy(nn_inp)
            if self.cuda:
                nn_inp = nn_inp.cuda()
            nn_outp = D(nn_inp).cpu().detach().numpy()
            if D.mode == "vaegan":
                nn_outp = nn_outp[0]
            contour = plt.contourf(
                xx,
                yy,
                nn_outp.reshape((self.d_output_resolution, self.d_output_resolution)),
                cmap="Greys"
            )
            plt.colorbar(contour)
        Gx.eval()
        reals = self.data.cpu().numpy()
        gens = Gx(self.static_z).cpu().detach().numpy()
        plt.scatter(reals[:, 0], reals[:, 1], color="red", s=3, alpha=0.1, label="x")
        plt.scatter(gens[:, 0], gens[:, 1], color="blue", s=3, alpha=0.2, label="Gx(z)")
        if self.output_reproductions:
            Gz.eval()
            x = self.data
            if self.cuda:
                x = x.cuda()
            # Gz (hopefully) outputs the 3-tuple sample, mean, log_variance. The following line grabs the right index.
            gz_index = 0 if self.sample_reconstructions else 1

            x_recon = Gx(Gz(x)[gz_index]).cpu().detach()
            plt.scatter(x_recon[:, 0], x_recon[:, 1], color="green", s=3, alpha=0.2, label="Gx(Gz(x))")
            Gz.train()

        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.savefig(os.path.join(self.path, "epoch_%04d.png"%state_dict["epoch"]))
        Gx.train()
