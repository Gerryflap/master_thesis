from trainloops.train_loop import TrainLoop
import torch


class EncBasedDecTrainLoop(TrainLoop):
    def __init__(
            self,
            listeners: list,
            generator,
            enc,
            dec_discriminator,
            gen_optimizer,
            discr_optimizer,
            pretrain_epochs,
            dataloader: torch.utils.data.DataLoader,
            cuda=False,
            epochs=1,
            beta=1.0,
            latent_discr=None,
            latent_discr_optimizer=None,
    ):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.enc = enc
        self.dec_discriminator = dec_discriminator
        self.generator = generator
        self.gen_optimizer = gen_optimizer
        self.discr_optimizer = discr_optimizer
        self.dataloader = dataloader
        self.cuda = cuda
        self.beta = beta
        self.pretrain_epochs = pretrain_epochs
        self.latent_discr = latent_discr
        self.latent_discr_optimizer = latent_discr_optimizer
        if self.latent_discr is None:
            self.latent_discr = torch.nn.Sequential(
                torch.nn.Linear(self.enc.latent_size, max(self.enc.latent_size * 2, 256), bias=True),
                torch.nn.LeakyReLU(0.02),

                torch.nn.Linear(max(self.enc.latent_size * 2, 256), max(self.enc.latent_size * 2, 256), bias=False),
                torch.nn.LeakyReLU(0.02),

                torch.nn.Linear(max(self.enc.latent_size * 2, 256), 1, bias=False),
                torch.nn.Sigmoid()
            )

        if self.latent_discr_optimizer is None:
            self.latent_discr_optimizer = torch.optim.Adam(self.latent_discr.parameters(), 0.0001, betas=(0.5, 0.999))

    def epoch(self):
        self.generator.train()
        self.enc.train()
        self.dec_discriminator.train()
        self.latent_discr.train()

        for i, (mbatch, _) in enumerate(self.dataloader):
            if mbatch.size()[0] != self.batch_size:
                continue

            if self.cuda:
                mbatch = mbatch.cuda()

            # Compute D loss
            z_fake, means, log_vars = self.enc(mbatch)
            x_recon = self.dec_discriminator(z_fake)
            loss_d = torch.nn.functional.mse_loss(x_recon, mbatch)

            pred_latent_d = self.latent_discr(z_fake)
            loss_enc_latent = torch.nn.functional.binary_cross_entropy(
                pred_latent_d, torch.ones_like(pred_latent_d))
            loss_d = loss_d + self.beta * loss_enc_latent

            self.discr_optimizer.zero_grad()
            # Backpropagate the errors
            loss_d.backward()

            # Update parameters
            self.discr_optimizer.step()

            # Train the latent discriminator
            z_fake = z_fake.detach()
            pred_fake = self.latent_discr(z_fake)

            z_real = self.generate_z_batch(self.batch_size)
            z_real.requires_grad = True
            pred_real = self.latent_discr(z_real)

            loss_fake = torch.nn.functional.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
            loss_real = torch.nn.functional.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
            pred_real_scalar = pred_real.sum()

            real_grad = torch.autograd.grad(pred_real_scalar, z_real, create_graph=True)[0]
            loss_grad = torch.pow(real_grad, 2).sum()

            loss_d_latent = loss_fake + loss_real + 10 * loss_grad

            self.latent_discr_optimizer.zero_grad()
            loss_d_latent.backward()
            self.latent_discr_optimizer.step()

            if self.current_epoch >= self.pretrain_epochs:

                # Compute G loss
                z_g_input = self.generate_z_batch(self.batch_size)
                x_gen = self.generator(z_g_input)
                z_pred, z_pred_mean, _ = self.enc(x_gen)

                loss_g = torch.nn.functional.l1_loss(z_pred_mean, z_g_input, reduction='mean')

                # Update generator
                self.gen_optimizer.zero_grad()
                loss_g.backward()
                self.gen_optimizer.step()
            else:
                loss_g = torch.zeros((1,))
        return {
            "epoch": self.current_epoch,
            "losses": {
                "loss_g": loss_g.detach().item(),
                "loss_d": loss_d.detach().item(),
                "loss_d_latent": loss_d_latent.detach().item(),
                "loss_enc_latent": loss_enc_latent.detach().item()
            },
            "networks": {
                # Use the dec discriminator as generator when pretraining
                "G": self.generator if self.current_epoch >= self.pretrain_epochs else self.dec_discriminator,
                "enc": self.enc,
                "dec_discriminator": self.dec_discriminator,
                "d_latent": self.latent_discr
            },
            "optimizers": {
                "gen_optimizer": self.gen_optimizer,
                "discr_optimizer": self.discr_optimizer,
            }
        }

    def generate_z_batch(self, batch_size):
        z = torch.normal(torch.zeros((batch_size, self.generator.latent_size)), 1)
        if self.cuda:
            z = z.cuda()
        return z

    def generate_batch(self, batch_size):
        # Generate random latent vectors
        z = self.generate_z_batch(batch_size)

        # Return outputs
        return self.generator(z)

    @staticmethod
    def loss_fn(x_recon, x, means, log_vars, beta=1.0):
        # l_recon = torch.nn.functional.binary_cross_entropy((x_recon + 1)/2, (x + 1)/2, reduction='sum')
        l_recon = torch.nn.functional.mse_loss(x_recon, x, reduction='sum')

        # The Dkl between a standard normal and the output distributions of the network
        l_prior = -0.5 * torch.sum(1 + log_vars - torch.pow(means, 2) - torch.exp(log_vars))

        return l_recon + beta * l_prior
