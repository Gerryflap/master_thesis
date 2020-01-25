import torch
from torch.utils.data import DataLoader

from trainloops.listeners.listener import Listener


class FRSAccuracyPrinter(Listener):
    def __init__(self, cuda, dataset):
        super().__init__()
        self.cuda = cuda
        self.dataloader = DataLoader(dataset, 32, drop_last=False)

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

        pos_acc = (pos_distances < 0.6).mean()
        neg_acc = (neg_distances >= 0.6).mean()
        print("Positive samples accuracy: ", pos_acc.detach().item())
        print("Negative samples accuracy: ", neg_acc.detach().item())
