from trainloops.train_loop import TrainLoop
import torch


def get_log_odds(raw_marginals, use_sigmoid):
    if use_sigmoid:
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    else:
        # Correct for normalization between -1 and 1
        raw_marginals = (raw_marginals + 1)/2
        marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))

class WCycleGanTrainLoop(TrainLoop):
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
            lambd_x=10.0,
            lambd_z=0.1,
            alpha=1.0
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
        self.lambd_x = lambd_x
        self.lambd_z = lambd_z
        self.alpha = alpha


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

            if i%self.D_steps_per_G_step == 0:
                # Compute reconstruction loss for z
                z_recon = self.Gz.encode(x_tilde)
                # L_recon = torch.nn.functional.mse_loss(z_recon, z).mean()
                L_recon = torch.nn.functional.l1_loss(z_recon, z).mean()

                # Compute losses for G
                gz_wgan_loss = -Dz_fake.mean()
                gx_wgan_loss = -Dx_fake.mean()

                g_loss = gz_wgan_loss + gx_wgan_loss + self.alpha * L_recon

            # Compute losses for D
            dx_wgan_loss = Dx_fake - Dx_real
            dz_wgan_loss = Dz_fake - Dz_real

            # creates an array of (batch_size, 1, ..., 1) with the same length as the size of x.
            # So (bs, 1) for 2d and (bs, 1, 1, 1) for 4d.
            size = [s if i == 0 else 1 for i, s in enumerate(x.size())]
            grad_outputs = torch.ones_like(Dx_real)

            epsx = torch.rand(size)
            if self.cuda:
                epsx = epsx.cuda()
            x_hat = epsx * x + (1.0-epsx) * x_tilde.detach()
            x_hat.requires_grad = True
            Dx_hat = self.Dx(x_hat)
            grad = torch.autograd.grad(Dx_hat, x_hat, create_graph=True, only_inputs=True, grad_outputs=grad_outputs)[0]
            grad_norm = grad.norm(2, dim=list(range(1, len(grad.size()))))
            dx_grad_loss = torch.pow(grad_norm - 1, 2)

            epsz = torch.rand((self.batch_size, 1))
            if self.cuda:
                epsz = epsz.cuda()
            z_hat = epsz * z + (1.0-epsz) * z_tilde.detach()
            z_hat.requires_grad = True
            Dz_hat = self.Dz(z_hat)
            grad = torch.autograd.grad(Dz_hat, z_hat, create_graph=True, only_inputs=True, grad_outputs=grad_outputs)[0]
            grad_norm = grad.norm(2, dim=list(range(1, len(grad.size()))))
            dz_grad_loss = torch.pow(grad_norm - 1, 2)

            dx_loss = (dx_wgan_loss + self.lambd_x * dx_grad_loss).mean()
            dz_loss = (dz_wgan_loss + self.lambd_z * dz_grad_loss).mean()

            d_loss = dx_loss + dz_loss

            if i%self.D_steps_per_G_step == 0:
                self.G_optimizer.zero_grad()
                g_loss.backward(retain_graph=True)
                self.G_optimizer.step()

            self.D_optimizer.zero_grad()
            d_loss.backward()
            self.D_optimizer.step()



        return {
            "epoch": self.current_epoch,
            "losses": {
                "Dx_loss": dx_loss.detach().item(),
                "Dz_loss": dz_loss.detach().item(),

                "Gx_wgan_loss": gx_wgan_loss.detach().item(),
                "Gz_wgan_loss": gz_wgan_loss.detach().item(),
                "G_loss": g_loss.detach().item(),
                "D_loss": d_loss.detach().item(),
            },
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

