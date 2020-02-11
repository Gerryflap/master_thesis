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

class GanTrainLoop(TrainLoop):
    def __init__(
            self,
            listeners: list,
            G,
            D,
            G_optimizer,
            D_optimizer,
            dataloader: torch.utils.data.DataLoader,
            D_steps_per_G_step=1,
            cuda=False,
            epochs=1,
            lambd=10.0
    ):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.dataloader = dataloader
        self.D_steps_per_G_step = D_steps_per_G_step
        self.cuda = cuda
        self.lambd = lambd

    def epoch(self):

        for i, (real_batch, _) in enumerate(self.dataloader):
            if real_batch.size()[0] != self.batch_size:
                continue

            if self.cuda:
                real_batch = real_batch.cuda()

            if self.current_epoch == 0 and i == 0:
                    if hasattr(self.G, 'output_bias'):
                        self.G.output_bias.data = get_log_odds(real_batch, use_sigmoid=True)
                    else:
                        print("WARNING! Gx does not have an \"output_bias\". "
                              "Using untied biases as the last layer of Gx is advised!")
            # Train D

            # Make gradients for D zero
            self.D.zero_grad()

            # Compute outputs for real images
            d_real_outputs = self.D(real_batch)

            # Generate a fake image batch
            fake_batch = self.generate_batch(self.batch_size).detach()

            # Compute outputs for fake images
            d_fake_outputs = self.D(fake_batch)

            # Compute losses
            d_loss = (d_fake_outputs - d_real_outputs)

            size = [s if i == 0 else 1 for i, s in enumerate(fake_batch.size())]
            eps = torch.rand(size)
            if self.cuda:
                eps = eps.cuda()
            x_hat = eps * real_batch + (1.0-eps) * fake_batch
            x_hat.requires_grad = True
            dis_out = self.D(x_hat)
            grad_outputs = torch.ones_like(dis_out)
            grad = torch.autograd.grad(dis_out, x_hat, create_graph=True, only_inputs=True, grad_outputs=grad_outputs)[0]
            grad_norm = grad.norm(2, dim=list(range(1, len(grad.size()))))
            d_grad_loss = torch.pow(grad_norm - 1, 2)

            d_loss = d_loss + self.lambd * d_grad_loss
            d_loss = d_loss.mean()

            d_loss.backward()

            # Update weights
            self.D_optimizer.step()

            if i % self.D_steps_per_G_step == 0:
                # Train G (this is sometimes skipped to balance G and D according to the d_steps parameter)

                # Make gradients for G zero
                self.G.zero_grad()

                # Generate a batch of fakes
                fake_batch = self.generate_batch(self.batch_size)

                # Compute loss for G, images should become more 'real' to the discriminator
                g_loss = -self.D(fake_batch).mean()
                g_loss.backward()

                self.G_optimizer.step()



        return {
            "epoch": self.current_epoch,
            "losses": {
                "D_loss": d_loss.detach().item(),
                "G_loss": g_loss.detach().item(),
                "Mean grad norm": grad_norm.mean().detach().item(),
            },
            "networks": {
                "G": self.G,
                "D": self.D,
            },
            "optimizers": {
                "G_optimizer": self.G_optimizer,
                "D_optimizer": self.D_optimizer
            }
        }

    def generate_z_batch(self, batch_size):
        z = torch.normal(torch.zeros((batch_size, self.G.latent_size)), 1)
        if self.cuda:
            z = z.cuda()
        return z

    def generate_batch(self, batch_size):
        # Generate random latent vectors
        z = self.generate_z_batch(batch_size)

        # Return outputs
        return self.G(z)
