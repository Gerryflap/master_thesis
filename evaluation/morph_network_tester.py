import argparse
import torch
import matplotlib.pyplot as plt

from util.interpolation import torch_slerp

parser = argparse.ArgumentParser(description="Image to latent vector converter.")
parser.add_argument("--enc", required=True, action="store", type=str, help="Path to Gz/Encoder model")
parser.add_argument("--n_samples", action="store", type=str, help="Number of z morphs to generate", default=1000)
parser.add_argument("--use_z_morph_sort", action="store_true", help="Sorts z1 and puts all others in that order as well")

args = parser.parse_args()

Gz = torch.load(args.enc).cpu()

latent_size = Gz.latent_size

z1 = torch.randn((args.n_samples, latent_size))
z2 = torch.randn((args.n_samples, latent_size))

z_morph = Gz.morph_zs(z1, z2)
linear_morph = 0.5*(z1 + z2)
slerp_morph = torch_slerp(0.5, z1, z2)

z_morph_l2s, z_sort = z_morph.norm(2, 1).detach().sort()
if args.use_z_morph_sort:
    z1_l2s = z1.norm(2, 1)[z_sort].detach()
    z2_l2s = z2.norm(2, 1)[z_sort].detach()
    linear_morph_l2s = linear_morph.norm(2, 1)[z_sort].detach()
    dist_z_morph_to_linear = (linear_morph - z_morph)[z_sort].norm(2, 1).detach()
    slerp_morph_l2s = slerp_morph.norm(2, 1)[z_sort].detach()
else:
    z1_l2s = z1.norm(2, 1).detach().sort()[0]
    z2_l2s = z2.norm(2, 1).detach().sort()[0]
    z_morph_l2s = z_morph.norm(2, 1).detach().sort()[0]
    linear_morph_l2s = linear_morph.norm(2, 1).detach().sort()[0]
    dist_z_morph_to_linear = (linear_morph - z_morph).norm(2, 1).detach().sort()[0]
    slerp_morph_l2s = slerp_morph.norm(2, 1).detach().sort()[0]


plt.plot(z1_l2s.numpy(), label="z1")
plt.plot(z2_l2s.numpy(), label="z2")
plt.plot(z_morph_l2s.numpy(), label="z_morph")
plt.plot(linear_morph_l2s.numpy(), label="linear morph")
# plt.plot(slerp_morph_l2s.numpy(), label="slerp morph")
# plt.plot(dist_z_morph_to_linear, label="L2 dist between linear and z_morph")
plt.title("Sorted $L_2$ norms of latent vector distributions")
plt.ylabel("$||z||_2$")
plt.xlabel("Index in sorted list")
plt.legend()
plt.show()