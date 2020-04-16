import torch

from util.torch.initialization import weights_init


class MorphNetwork(torch.nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.morph_net = torch.nn.Sequential(
            torch.nn.Linear(latent_size * 2, latent_size * 2),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(latent_size * 2, latent_size * 2),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(latent_size * 2, latent_size),
        )

    def forward(self, zs):
        z1, z2 = zs
        return self.morph_zs(z1, z2)

    def morph_zs(self, z1, z2):
        z = self.morph_net(torch.cat([z1, z2], dim=1))
        return z

    def pretrain_morph_network(self, pretrain_opt, n_batches=10000, batch_size=64):
        print("Pretraining morph network")
        device = self.morph_net.parameters().__next__().get_device()
        for i in range(n_batches):
            z1 = torch.randn((batch_size, self.latent_size), device=device)
            z2 = torch.randn((batch_size, self.latent_size), device=device)

            z_target = 0.5 * (z1 + z2)
            pred = self.morph_zs(z1, z2)
            loss = torch.nn.functional.mse_loss(pred, z_target)
            pretrain_opt.zero_grad()
            loss.backward()
            pretrain_opt.step()
            if i == 0:
                print("Loss on first pretrain batch: ", loss.detach().item())
        print("Pretraining done...")
        print("Final pretrain loss: %f" % loss.detach().item())


    def init_weights(self):
        self.apply(weights_init)