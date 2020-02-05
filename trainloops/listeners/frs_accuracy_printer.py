import math

import torch
from torch.utils.data import DataLoader

from trainloops.listeners.listener import Listener


class FRSAccuracyPrinter(Listener):
    def __init__(self, cuda, dataset, margin=None, far=0.05):
        """
        Initializes the FRS Accuracy printer
        :param cuda: Whether to use cuda
        :param dataset: The dataset to use for evaluation.
            Keep in mind that the WHOLE dataset is used. This is done in batches of 32
        :param margin: If specified, overrides far and uses the specified margin as the threshold to determine real from fake
        :param far: False Accept Rate. Aims to get at most this FAR. The threshold is set such that this is reached.
        """
        super().__init__()
        self.cuda = cuda
        self.dataloader = DataLoader(dataset, 32, drop_last=False)
        self.margin = margin
        self.far = far


    def initialize(self):
        pass

    def report(self, state_dict):
        print("Epoch: %d"%state_dict["epoch"])
        model = state_dict["networks"]["model"]
        model.eval()

        anchors = []
        positives = []
        negatives = []

        for anchor, pos, neg in self.dataloader:
            if self.cuda:
                anchor = anchor.cuda()
                pos = pos.cuda()
                neg = neg.cuda()

            z_anchor = model(anchor)
            z_pos = model(pos)
            z_neg = model(neg)

            anchors.append(z_anchor.cpu().detach())
            positives.append(z_pos.cpu().detach())
            negatives.append(z_neg.cpu().detach())

        anchors = torch.cat(anchors, dim=0)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)

        pos_distances = torch.sqrt(torch.sum(torch.pow(anchors - positives, 2), dim=1))
        neg_distances = torch.sqrt(torch.sum(torch.pow(anchors - negatives, 2), dim=1))

        threshold = self.margin
        if threshold is None:
            sorted_index = min((len(self.dataloader), math.floor(self.far*len(self.dataloader))))
            neg_distances_sorted, _ = torch.sort(neg_distances)
            threshold = neg_distances_sorted[sorted_index]

        pos_acc = (pos_distances < threshold).type(torch.float32).mean()
        neg_acc = (neg_distances >= threshold).type(torch.float32).mean()
        print("Positive samples accuracy: ", pos_acc.detach().item())
        print("Negative samples accuracy: ", neg_acc.detach().item())
        print("Positive samples mean distance: ", pos_distances.mean().detach().item())
        print("Negative samples mean distance: ", neg_distances.mean().detach().item())
