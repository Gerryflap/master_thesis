"""
    This trainloop trains a morph network using already trained Gx, Gz and D networks
"""
import torch
import torch.nn.functional as F

from trainloops.train_loop import TrainLoop
from util.interpolation import torch_slerp
from util.torch.losses import euclidean_distance


class MorphNetTrainLoop(TrainLoop):
    def __init__(self, listeners: list, morph_net, Gz, Gx, D, optim, dataloader, cuda=False, epochs=1,
                 morph_loss_factor=1.0, morph_loss_mode="pixelwise",
                 frs_model=None):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.morph_net = morph_net
        self.Gz = Gz
        self.Gx = Gx
        self.D = D
        self.optim = optim
        self.dataloader = dataloader
        self.cuda = cuda
        self.frs_model = frs_model

        if morph_loss_mode not in ["pixelwise", "dis_l", "frs"]:
            raise ValueError("Reconstruction loss mode must be one of \"pixelwise\", \"dis_l\" or \"frs\"")
        self.morph_loss_mode = morph_loss_mode
        self.morph_loss_factor = morph_loss_factor

        self.Gx.requires_grad = False
        self.Gz.requires_grad = False
        self.D.requires_grad = False
        self.Gx.eval()
        self.Gz.eval()
        self.D.eval()


    def epoch(self):
        for i, (x1, x2) in enumerate(self.dataloader):
            if x1.size(0) != self.batch_size:
                continue

            # Train D
            # Draw M (= batch_size) samples from dataset and prior. x samples are already loaded by dataloader
            if self.cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            # Sample from conditionals (sampling is implemented by models)
            z1_hat = self.Gz.encode(x1)
            z2_hat = self.Gz.encode(x2)

            z_morph = self.morph_net((z1_hat, z2_hat))

            x_morph = self.Gx(z_morph)

            if self.morph_loss_mode == "pixelwise":
                L_morph = self.morph_loss(x_morph, x1, x2, 0.5)
            elif self.morph_loss_mode == "dis_l":
                _, dis_l_morph = self.D.compute_dx(x_morph)
                _, dis_l_x2 = self.D.compute_dx(x2)
                _, dis_l_x1 = self.D.compute_dx(x1)
                L_morph = self.morph_loss(dis_l_morph, dis_l_x1, dis_l_x2, 0.5)
            else:
                dist1 = euclidean_distance(self.frs_model(x_morph), self.frs_model(x1))
                dist2 = euclidean_distance(self.frs_model(x_morph), self.frs_model(x2))
                L_morph = torch.max(dist1, dist2)


            # Regularizations to constrain the outputs of the morph network
            z_in = self.generate_z_batch(self.batch_size)
            z_out = self.Gz.morph_zs(z_in, z_in)
            morph_cons_loss = torch.nn.functional.mse_loss(z_out, z_in)

            # Added 9 April 2020
            z_morph_detached_from_Gz = self.morph_net((z1_hat.detach(), z2_hat.detach()))
            l2_norms = torch.max(z1_hat.detach().norm(2, dim=1), z1_hat.detach().norm(2, dim=1))
            morph_l2_norms = z_morph_detached_from_Gz.norm(2, dim=1)
            morph_scale_loss = (torch.nn.functional.relu(morph_l2_norms - l2_norms)**2).mean()
            morph_cons_loss += 10.0 * morph_scale_loss

            # ========== Back propagation and updates ==========

            self.optim.zero_grad()
            L_morph.backward(retain_graph=True)
            morph_cons_loss.backward()
            self.optim.step()

        losses = {
                "L_morph": L_morph.detach().item(),
                "L_regularizations": morph_cons_loss.detach().item(),
            }

        return {
            "epoch": self.current_epoch,
            "losses": losses,
            "networks": {
                "morph_net": self.morph_net,
                "Gx": self.Gx,
                "Gz": self.Gz,
                "D": self.D,
            },
            "optimizers": {
                "optimizer": self.optim,
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

    def reconstruction_loss(self, x_recon, target):
        if self.reconstruction_loss_mode == "pixelwise":
            return self.morgan_pixel_loss(x_recon, target).mean()
        else:
            return self.dis_l_loss(x_recon, target).mean()

    def morph_loss(self, x_morph, x1, x2, beta):
        if self.morph_loss_mode == "pixelwise":
            x1_loss = self.morgan_pixel_loss(x_morph, x1)
            x2_loss = self.morgan_pixel_loss(x_morph, x2)
            return (beta * x1_loss + (1-beta) * x2_loss).mean()
        else:
            x1_loss = self.dis_l_loss(x_morph, x1)
            x2_loss = self.dis_l_loss(x_morph, x2)
            return (beta * x1_loss + (1-beta) * x2_loss).mean()

    def dis_l_loss(self, dis_l_prediction, dis_l_target):
        return torch.nn.functional.mse_loss(dis_l_prediction, dis_l_target, reduction='none').view(self.batch_size, -1).mean(dim=1, keepdim=True)

    @staticmethod
    def morgan_pixel_loss(x_recon, target):
        absolute_errors = torch.abs(x_recon - target)
        # WxH = float(int(absolute_errors.size()[2]) * int(absolute_errors.size()[3]))
        # loss = absolute_errors.sum()/WxH
        loss = absolute_errors.view(x_recon.size(0), -1).mean(dim=1, keepdim=True)
        return loss

    def add_instance_noise(self, x):
        noise_factor = self.d_img_noise_std * \
                       (1 if not self.decrease_noise else 1 - (self.current_epoch / self.epochs))
        return x + torch.randn_like(x) * noise_factor
