"""
    Models a train loop for ALI: Adversarially Learned Inference (https://arxiv.org/abs/1606.00704)
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop


class ALITrainLoop(TrainLoop):
    def __init__(self, listeners: list, Gz, Gx, D, optim_G, optim_D, dataloader, cuda=False, epochs=1):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.Gz = Gz
        self.Gx = Gx
        # self.G = torch.nn.ModuleList([self.Gx, self.Gz])
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.dataloader = dataloader
        self.cuda = cuda

    def epoch(self):
        self.Gx.train()
        self.Gz.train()
        self.D.train()

        for i, (x, _) in enumerate(self.dataloader):
            if x.size()[0] != self.batch_size:
                continue
            # Draw M (= batch_size) samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x = x.cuda()
            z = self.generate_z_batch(self.batch_size)

            # Sample from conditionals (sampling is implemented by models)
            z_hat = self.Gz.encode(x)
            x_tilde = self.Gx(z)

            # Compute discriminator predictions
            dis_q = self.D((x, z_hat))
            dis_p = self.D((x_tilde, z))

            # Compute Discriminator loss
            L_d = F.binary_cross_entropy_with_logits(dis_q, torch.ones_like(dis_q), reduction="mean") + \
                  F.binary_cross_entropy_with_logits(dis_p, torch.zeros_like(dis_q), reduction="mean")

            L_gz = F.binary_cross_entropy_with_logits(dis_q, torch.zeros_like(dis_q), reduction="mean")
            L_gx = F.binary_cross_entropy_with_logits(dis_p, torch.ones_like(dis_q), reduction="mean")
            L_g = L_gz + L_gx

            # Gradient update on Discriminator network
            self.optim_D.zero_grad()
            L_d.backward(create_graph=True)
            self.optim_D.step()

            # Gradient update on Generator networks
            self.optim_G.zero_grad()
            L_g.backward()
            self.optim_G.step()
        self.Gx.eval()
        self.Gz.eval()
        self.D.eval()
        return {
            "epoch": self.current_epoch,
            "losses": {
                "D_loss: ": L_d.detach().item(),
                "G_loss: ": L_g.detach().item(),
            },
            "networks": {
                "Gx": self.Gx,
                "Gz": self.Gz,
                "D": self.D,
            },
            "optimizers": {
                "G_optimizer": self.optim_G,
                "D_optimizer": self.optim_D
            }
        }

    def generate_z_batch(self, batch_size):
        z = torch.normal(torch.zeros((batch_size, self.Gx.latent_size)), 1)
        if self.cuda:
            z = z.cuda()
        return z

    def generate_batch(self, batch_size):
        # Generate random latent vectors
        z = self.generate_z_batch(batch_size)

        # Return outputs
        return self.Gx(z)
