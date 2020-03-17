from trainloops.train_loop import TrainLoop
import torch
import torch.nn.functional as F

class GanTrainLoop(TrainLoop):
    def __init__(
            self,
            listeners: list,
            Gx,
            Gz,
            Dx,                         # Dx should be recon discriminator
            Dz,
            G_optimizer,
            D_optimizer,
            dataloader: torch.utils.data.DataLoader,
            D_steps_per_G_step=2,
            cuda=False,
            epochs=1
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

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.real_label = torch.zeros((self.batch_size, 1))
        self.fake_label = torch.ones((self.batch_size, 1))

        if cuda:
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()

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
            d_real_loss = self.loss_fn(d_real_outputs, self.real_label)
            d_real_loss.backward()

            # Generate a fake image batch
            fake_batch = self.generate_batch(self.batch_size).detach()

            # Compute outputs for fake images
            d_fake_outputs = self.D(fake_batch)
            d_fake_loss = self.loss_fn(d_fake_outputs, self.fake_label)
            d_fake_loss.backward()

            # Compute loss outputs
            d_loss = d_fake_loss + d_real_loss

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)

            # Update weights
            self.D_optimizer.step()

            if i % self.D_steps_per_G_step == 0:
                # Train G (this is sometimes skipped to balance G and D according to the d_steps parameter)

                # Make gradients for G zero
                self.G.zero_grad()

                # Generate a batch of fakes
                fake_batch = self.generate_batch(self.batch_size)

                # Compute loss for G, images should become more 'real' to the discriminator
                g_loss = self.loss_fn(self.D(fake_batch), self.real_label)
                g_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)

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
