import torch
from models.conv64_ali.encoder import Encoder64


class EncoderMorphNet64(Encoder64):
    def __init__(self, latent_size, h_size, use_mish=False, n_channels=1, deterministic=False, cap_variance=True,
                 use_lr_norm=False, block_Gz_morph_grads=False):
        super().__init__(latent_size, h_size, use_mish, n_channels, deterministic, cap_variance, use_lr_norm)

        self.morph_net = torch.nn.Sequential(
            torch.nn.Linear(latent_size*2, latent_size*2),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(latent_size * 2, latent_size * 2),
            torch.nn.LeakyReLU(0.02),

            torch.nn.Linear(latent_size * 2, latent_size),
        )
        self.block_Gz_morph_grads = block_Gz_morph_grads

    def morph_zs(self, z1, z2):
        if self.block_Gz_morph_grads:
            z1 = z1.detach()
            z2 = z2.detach()
        z = self.morph_net(torch.cat([z1, z2], dim=1))
        return z

    def pretrain_morph_network(self, n_batches=10000, batch_size=64):
        print("Pretraining morph network")
        pretrain_opt = torch.optim.Adam(self.morph_net.parameters(), 0.01)
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
        print("Final pretrain loss: %d"%loss.detach().item())