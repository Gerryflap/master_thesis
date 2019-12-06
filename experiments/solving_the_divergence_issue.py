"""
    Investigation into the divergence behavior of certain models
"""
import torch


def v(x):
    return x.detach().item()


folder_path = "results/celeba64/epoch_000070/"

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