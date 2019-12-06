"""
    Investigation into the divergence behavior of certain models
"""
import torch


def v(x):
    return x.detach().item()


folder_path = "results/celeba28/MorGAN/2019-12-06T16:04:51/params/all_epochs/"

Gx = torch.load(folder_path + "Gx.pt").cpu()
D = torch.load(folder_path + "D.pt").cpu()

Gx.train()
D.train()

print("Gx parameter statistics:")
for param in Gx.parameters():
    print(param.size(), v(param.mean()), v(param.var()))
print()

print("D parameter statistics:")
for param in D.parameters():
    print(param.size(), v(param.mean()), v(param.var()))
print()

z = torch.normal(0, 1, (64, 256))
x = Gx(z)
dx = D((x, z))

print(dx.detach())