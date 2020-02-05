from trainloops.train_loop import TrainLoop
import torch
import torch.nn.functional as F

class GanTrainLoop(TrainLoop):
    def __init__(
            self,
            listeners: list,
            G,
            D,
            G_optimizer,
            D_optimizer,
            dataloader: torch.utils.data.DataLoader,
            D_steps_per_G_step=2,
            cuda=False,
            epochs=1
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

    def epoch(self):

        for i, (real_batch, _) in enumerate(self.dataloader):
            if real_batch.size()[0] != self.batch_size:
                continue
            # Train D

            # Make gradients for D zero
            self.D.zero_grad()
            if self.cuda:
                real_batch = real_batch.cuda()

            # Compute outputs for real images
            d_real_outputs = self.D(real_batch)

            # Generate a fake image batch
            fake_batch = self.generate_batch(self.batch_size).detach()

            # Compute outputs for fake images
            d_fake_outputs = self.D(fake_batch)

            # Compute losses
            d_loss = (d_fake_outputs - d_real_outputs).mean()

            eps = torch.randn((self.batch_size, 1, 1, 1))
            if self.cuda:
                eps = eps.cuda()
            x_hat = eps * real_batch + (1.0-eps) * fake_batch
            grad = torch.autograd.grad(self.D(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]
            d_grad_loss = (torch.pow(grad, 2) - 1).mean()

            d_loss = d_loss + 10.0 * d_grad_loss

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
