import torch


def euclidean_distance_per_element(x1, x2):
    return torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), dim=1))


def euclidean_distance(x1, x2):
    return euclidean_distance_per_element(x1, x2).mean()
