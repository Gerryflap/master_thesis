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
            D_steps_per_G_step=1,
            cuda=False,
            epochs=1,
            E=None,
            E_optimizer=None,
            dis_l=False,
            r1_reg_gamma=0.0,
            compute_r1_every_n_steps=1
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
        self.r1_reg_gamma = r1_reg_gamma * compute_r1_every_n_steps
        self.compute_r1_every_n_steps = compute_r1_every_n_steps

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.real_label = torch.zeros((self.batch_size, 1))
        self.fake_label = torch.ones((self.batch_size, 1))

        if cuda:
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()

        self.dis_l = dis_l

        # If an encoder is specified, train the encoder to encode images into G's latent space
        self.E = E
        self.E_optimizer = E_optimizer

    def epoch(self):

        for i, (real_batch, _) in enumerate(self.dataloader):
            if real_batch.size()[0] != self.batch_size:
                continue
            # Train D

            # Make gradients for D zero
            self.D.zero_grad()
            if self.cuda:
                real_batch = real_batch.cuda()

            if self.r1_reg_gamma != 0 and i % self.compute_r1_every_n_steps == 0:
                real_batch.requires_grad = True

            # Compute outputs for real images
            d_real_outputs = self.D(real_batch)
            if self.r1_reg_gamma == 0:
                d_real_loss = self.loss_fn(d_real_outputs, self.real_label)
            else:
                d_real_loss = torch.nn.functional.softplus(-d_real_outputs).mean()

            # Generate a fake image batch
            fake_batch = self.generate_batch(self.batch_size).detach()

            # Compute outputs for fake images
            d_fake_outputs = self.D(fake_batch)
            if self.r1_reg_gamma == 0:
                d_fake_loss = self.loss_fn(d_fake_outputs, self.fake_label)
            else:
                d_fake_loss = torch.nn.functional.softplus(d_fake_outputs).mean()


            # Compute loss outputs
            d_loss = d_fake_loss + d_real_loss

            if self.r1_reg_gamma != 0 and i % self.compute_r1_every_n_steps == 0:
                # Computes an R1-like loss
                grad_outputs = torch.ones_like(d_real_outputs)
                x_grads = torch.autograd.grad(
                    d_real_outputs,
                    real_batch,
                    create_graph=True,
                    only_inputs=True,
                    grad_outputs=grad_outputs
                )[0]
                r1_loss = x_grads.norm(2, dim=list(range(1, len(x_grads.size())))).mean()
                d_loss += (self.r1_reg_gamma / 2.0) * r1_loss

            d_loss.backward()

            # Update weights
            self.D_optimizer.step()
            real_batch.requires_grad = False

            if i % self.D_steps_per_G_step == 0:
                # Train G (this is sometimes skipped to balance G and D according to the d_steps parameter)

                # Make gradients for G zero
                self.G.zero_grad()

                # Generate a batch of fakes
                fake_batch = self.generate_batch(self.batch_size)

                # Compute loss for G, images should become more 'real' to the discriminator
                if self.r1_reg_gamma == 0:
                    g_loss = self.loss_fn(self.D(fake_batch), self.real_label)
                else:
                    g_loss = torch.nn.functional.softplus(-self.D(fake_batch)).mean()
                g_loss.backward()

                self.G_optimizer.step()

                # If an encoder is specified, train it
                if self.E is not None:
                    self.E.zero_grad()
                    z = self.E.encode(real_batch)
                    self.G.requires_grad = False
                    x_recon = self.G(z)

                    if self.dis_l:
                        disl_x = self.D.compute_disl(real_batch)
                        disl_xrecon = self.D.compute_disl(x_recon)
                        encoder_loss = torch.nn.functional.l1_loss(disl_xrecon, disl_x)
                    else:
                        encoder_loss = torch.nn.functional.l1_loss(x_recon, real_batch)

                    encoder_mean_reg = torch.pow(z.mean(), 2)
                    encoder_var_reg = torch.pow(z.var(dim=0).mean() - 1.0, 2)
                    encoder_loss += encoder_mean_reg + encoder_var_reg

                    encoder_loss.backward()
                    self.E_optimizer.step()

                    self.G.requires_grad = True

        losses = {
                "D_loss": d_loss.detach().item(),
                "G_loss": g_loss.detach().item(),
            }

        if self.r1_reg_gamma != 0.0:
            losses["r1_loss"] = r1_loss.detach().item()

        return {
            "epoch": self.current_epoch,
            "losses": losses,
            "networks": {
                "G": self.G,
                "D": self.D,
                "E": self.E
            },
            "optimizers": {
                "G_optimizer": self.G_optimizer,
                "D_optimizer": self.D_optimizer,
                "E_optimizer": self.E_optimizer,
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
