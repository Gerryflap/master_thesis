from trainloops.train_loop import TrainLoop
import torch
import torch.nn.functional as F


def get_log_odds(raw_marginals, use_sigmoid):
    if use_sigmoid:
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    else:
        # Correct for normalization between -1 and 1
        raw_marginals = (raw_marginals + 1) / 2
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


class MorGANDDTrainLoop(TrainLoop):
    def __init__(
            self,
            listeners: list,
            Gz,
            Gx,
            Dz,
            Dx,
            G_optimizer,
            D_optimizer,
            dataloader: torch.utils.data.DataLoader,
            D_steps_per_G_step=1,
            cuda=False,
            epochs=1,
            alpha_z=1.0,
            alpha_x=0.0,
            disable_D_limiting=False,
            use_dis_l_x_reconstruction_loss=False
    ):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.Gx = Gx
        self.Dx = Dx
        self.Gz = Gz
        self.Dz = Dz
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.dataloader = dataloader
        self.D_steps_per_G_step = D_steps_per_G_step
        self.cuda = cuda
        self.alpha_z = alpha_z
        self.alpha_x = alpha_x
        self.disable_D_limiting = disable_D_limiting
        self.use_dis_l_x_reconstruction_loss = use_dis_l_x_reconstruction_loss

    def epoch(self):

        for i, (x, _) in enumerate(self.dataloader):
            if x.size()[0] != self.batch_size:
                continue

            if self.cuda:
                x = x.cuda()

            if self.current_epoch == 0 and i == 0:
                if hasattr(self.Gx, 'output_bias'):
                    self.Gx.output_bias.data = get_log_odds(x, use_sigmoid=True)
                else:
                    print("WARNING! Gx does not have an \"output_bias\". "
                          "Using untied biases as the last layer of Gx is advised!")
            # Generate z and x_tilde
            z = self.generate_z_batch(self.batch_size)
            x_tilde = self.Gx(z)

            # Use x from the dataset to generate z_tilde
            z_tilde = self.Gz.encode(x)

            # Compute D values
            Dz_real = self.Dz(z)
            Dz_fake = self.Dz(z_tilde)

            Dx_real = self.Dx(x)
            Dx_fake = self.Dx(x_tilde)

            L_gx = F.binary_cross_entropy_with_logits(Dx_fake, torch.ones_like(Dx_real))
            L_gz = F.binary_cross_entropy_with_logits(Dz_fake, torch.ones_like(Dz_real))


            # Compute losses for D
            dx_loss = F.binary_cross_entropy_with_logits(Dx_real, torch.ones_like(Dx_real)) + \
                      F.binary_cross_entropy_with_logits(Dx_fake, torch.zeros_like(Dx_real))
            dz_loss = F.binary_cross_entropy_with_logits(Dz_real, torch.ones_like(Dz_real)) + \
                      F.binary_cross_entropy_with_logits(Dz_fake, torch.zeros_like(Dz_real))

            d_loss = dx_loss + dz_loss

            self.D_optimizer.zero_grad()

            if L_gx.detach().item() < 1.75 or self.disable_D_limiting:
                dx_loss.backward(retain_graph=True)

            if L_gz.detach().item() < 1.75 or self.disable_D_limiting:
                dz_loss.backward()

            self.D_optimizer.step()

            if i % self.D_steps_per_G_step == 0:
                # Train G
                # Generate z and x_tilde
                z = self.generate_z_batch(self.batch_size)
                x_tilde = self.Gx(z)

                # Use x from the dataset to generate z_tilde
                z_tilde = self.Gz.encode(x)

                # Compute D values
                Dz_fake = self.Dz(z_tilde)

                Dx_fake = self.Dx(x_tilde)

                # Compute reconstruction loss for z
                z_recon = self.Gz.encode(x_tilde)
                # L_recon = torch.nn.functional.mse_loss(z_recon, z).mean()
                L_recon_z = torch.nn.functional.mse_loss(z_recon, z)

                # Compute losses for G
                gx_loss = F.binary_cross_entropy_with_logits(Dx_fake, torch.ones_like(Dx_real))
                gz_loss = F.binary_cross_entropy_with_logits(Dz_fake, torch.ones_like(Dz_real))

                g_loss = gz_loss + gx_loss + self.alpha_z * L_recon_z
                if self.alpha_x != 0.0:
                    x_recon = self.Gx(z_tilde)
                    if self.use_dis_l_x_reconstruction_loss:
                        L_recon_x = self.dis_l_loss(x_recon, x)
                    else:
                        L_recon_x = torch.nn.functional.mse_loss(x_recon, x)
                    g_loss += self.alpha_x * L_recon_x

                self.G_optimizer.zero_grad()
                g_loss.backward()
                self.G_optimizer.step()

        losses = {
            "Dx_loss": dx_loss.detach().item(),
            "Dz_loss": dz_loss.detach().item(),

            "Gx_wgan_loss": gx_loss.detach().item(),
            "Gz_wgan_loss": gz_loss.detach().item(),
            "L_recon_z": L_recon_z.detach().item(),
            "G_loss": g_loss.detach().item(),
            "D_loss": d_loss.detach().item(),
        }

        if self.alpha_x != 0.0:
            losses["L_recon_x"] = L_recon_x.detach().item()

        return {
            "epoch": self.current_epoch,
            "losses": losses,
            "networks": {
                "Gx": self.Gx,
                "Gz": self.Gz,
                "Dx": self.Dx,
                "Dz": self.Dz,
            },
            "optimizers": {
                "G_optimizer": self.G_optimizer,
                "D_optimizer": self.D_optimizer
            }
        }

    def generate_z_batch(self, batch_size):
        z = torch.normal(torch.zeros((batch_size, self.Gx.latent_size)), 1)
        if self.cuda:
            z = z.cuda()
        return z

    def dis_l_loss(self, prediction, target):
        _, dis_l_prediction = self.Dx.compute_dx(prediction)
        _, dis_l_target = self.Dx.compute_dx(target)
        return torch.nn.functional.mse_loss(dis_l_prediction, dis_l_target)
