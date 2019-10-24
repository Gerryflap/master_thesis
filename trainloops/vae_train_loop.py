from trainloops.train_loop import TrainLoop
import torch


class VaeTrainLoop(TrainLoop):
    def __init__(
            self,
            listeners: list,
            enc,
            dec,
            enc_optimizer,
            dec_optimizer,
            dataloader: torch.utils.data.DataLoader,
            cuda=False,
            epochs=1
    ):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.enc = enc
        self.dec = dec
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self.dataloader = dataloader
        self.cuda = cuda

    def epoch(self):
        self.enc.train()
        self.dec.train()
        for i, (mbatch, _) in enumerate(self.dataloader):
            if mbatch.size()[0] != self.batch_size:
                continue

            if self.cuda:
                mbatch = mbatch.cuda()
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            zs, means, log_vars = self.enc(mbatch)

            x_recon = self.dec(zs)

            loss = self.loss_fn(x_recon, mbatch, means, log_vars)

            # Backpropagate the errors
            loss.backward()

            # Update parameters
            self.enc_optimizer.step()
            self.dec_optimizer.step()

        self.enc.eval()
        self.dec.eval()

        return {
            "epoch": self.current_epoch,
            "losses": {
                "loss: ": loss.detach().item(),
            },
            "networks": {
                "enc": self.enc,
                "dec": self.dec,
            },
            "optimizers": {
                "enc_optimizer": self.enc_optimizer,
                "dec_optimizer": self.dec_optimizer,
            }
        }

    def generate_z_batch(self, batch_size):
        z = torch.normal(torch.zeros((batch_size, self.dec.latent_size)), 1)
        if self.cuda:
            z = z.cuda()
        return z

    def generate_batch(self, batch_size):
        # Generate random latent vectors
        z = self.generate_z_batch(batch_size)

        # Return outputs
        return self.dec(z)

    @staticmethod
    def loss_fn(x_recon, x, means, log_vars):
        l_recon = torch.nn.functional.binary_cross_entropy((x_recon + 1)/2, (x + 1)/2, reduction='sum')

        # The Dkl between a standard normal and the output distributions of the network
        l_prior = -0.5 * torch.sum(1 + log_vars - torch.pow(means, 2) - torch.exp(log_vars))

        return l_recon + l_prior
