import torch

net = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    torch.nn.BatchNorm1d(1),
    torch.nn.Sigmoid()
)

net.train()

test = torch.normal(0, 1, (20, 2))
val = net(test)

for i in range(10):
    val2 = net(test)
    if (val2 != val).any():
        print("Unequality: ", val, val2)