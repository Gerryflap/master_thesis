import os

from trainloops.listeners.listener import Listener
import numpy as np
import torch
import matplotlib.pyplot as plt
import util


class MixtureVisualizer(Listener):
    def __init__(self, experiment_output_path, n_latent, valid_dataset, output_reproductions=False,
                 discriminator_output=False, d_output_resolution=100, cuda=False, sample_reconstructions=False,
                 every_n_epochs=10, generator_key=None, output_latent=False, output_grad_norm=False, ns_gan=False):
        super().__init__()
        folder_name = "mixture_outputs"
        if generator_key is not None:
            folder_name += "_" + generator_key
        self.path = os.path.join(experiment_output_path, "imgs", folder_name)
        util.output.make_result_dirs(self.path)
        if output_latent:
            util.output.make_result_dirs(os.path.join(self.path, "latent"))

        if output_grad_norm:
            util.output.make_result_dirs(os.path.join(self.path, "grad_norms"))
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
        self.generator_key = generator_key
        self.output_latent = output_latent
        self.output_grad_norm = output_grad_norm
        # Used for gradient comptuation:
        self.ns_gan = ns_gan

    def initialize(self):
        # Assume that it is a mixture dataset, hence we can just grab the data directly
        self.data = self.valid_dataset.data

    def report(self, state_dict):
        if state_dict["epoch"]%self.every_n_epochs != 0:
            return
        plt.clf()
        if self.generator_key is None:
            if "G" in state_dict["networks"]:
                Gx = state_dict["networks"]["G"]
            elif "dec" in state_dict["networks"]:
                Gx = state_dict["networks"]["dec"]
            elif "Gx" in state_dict["networks"]:
                Gx = state_dict["networks"]["Gx"]
            else:
                raise ValueError("Could not find a decoder-like network in the state dict!")
        else:
            Gx = state_dict["networks"][self.generator_key]

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

            nn_inp.requires_grad = True
            if D.mode == "ali":
                zs = Gz.encode(nn_inp)
                nn_outp = D((nn_inp, zs))
            else:
                nn_outp = D(nn_inp)
                if D.mode == "vaegan":
                    nn_outp = nn_outp[0]

            if self.output_grad_norm:
                plt.clf()
                if self.ns_gan:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(nn_outp, torch.ones_like(nn_outp))
                else:
                    loss = -torch.nn.functional.binary_cross_entropy_with_logits(nn_outp, torch.zeros_like(nn_outp))
                grads = torch.autograd.grad(loss, nn_inp, only_inputs=True)[0]
                grad_norms = grads.norm(2, dim=1).cpu().detach().numpy()
                contour = plt.contourf(
                    xx,
                    yy,
                    grad_norms.reshape((self.d_output_resolution, self.d_output_resolution)),
                    cmap="Greys"
                )
                plt.colorbar(contour)
                plt.xlim(-1.2, 1.2)
                plt.ylim(-1.2, 1.2)
                plt.savefig(os.path.join(self.path, "grad_norms", "epoch_%04d.png" % state_dict["epoch"]))

            nn_outp = nn_outp.cpu().detach().numpy()
            plt.clf()
            contour = plt.contourf(
                xx,
                yy,
                nn_outp.reshape((self.d_output_resolution, self.d_output_resolution)),
                cmap="Greys"
            )
            plt.colorbar(contour)
        # Gx.eval()
        reals = self.data.cpu().numpy()
        gens = Gx(self.static_z).cpu().detach().numpy()
        plt.scatter(reals[:, 0], reals[:, 1], color="red", s=3, alpha=0.1, label="x")
        plt.scatter(gens[:, 0], gens[:, 1], color="blue", s=3, alpha=0.2, label="Gx(z)")
        if self.output_reproductions:
            # Gz.eval()
            x = self.data
            if self.cuda:
                x = x.cuda()
            # Gz (hopefully) outputs the 3-tuple sample, mean, log_variance. The following line grabs the right index.
            gz_index = 0 if self.sample_reconstructions else 1
            z = Gz(x)[gz_index]
            x_recon = Gx(z).cpu().detach()
            plt.scatter(x_recon[:, 0], x_recon[:, 1], color="green", s=3, alpha=0.2, label="Gx(Gz(x))")
            # Gz.train()



        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.savefig(os.path.join(self.path, "epoch_%04d.png"%state_dict["epoch"]))
        # Gx.train()

        if self.output_reproductions and self.output_latent and z.size(1) == 2:
            print("Printing latent")
            plt.clf()
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            static_z = self.static_z.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            plt.title("Latent vector values")
            plt.scatter(static_z[:, 0], static_z[:, 1], color="red", s=3, alpha=0.1, label="z")
            plt.scatter(z[:, 0], z[:, 1], color="blue", s=3, alpha=0.2, label="Gz(x)")
            plt.savefig(os.path.join(self.path, "latent", "epoch_%04d.png"%state_dict["epoch"]))

