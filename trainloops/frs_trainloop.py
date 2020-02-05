from trainloops.train_loop import TrainLoop
import torch
import torch.nn.functional as F


class FRSTrainLoop(TrainLoop):
    def __init__(
            self,
            listeners: list,
            model,
            optimizer,
            dataloader: torch.utils.data.DataLoader,
            cuda=False,
            epochs=1,
            margin=0.6,
            use_hard_triplets=False,
    ):
        """
        Instantiates a FRS trainloop
        :param margin: Defines the margin used by the triplet loss
        :param use_hard_triplets: Uses hard triplets. This requires give_n_negatives > 1 in the triplets dataset
        """
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.cuda = cuda
        self.margin = margin
        self.use_hard_triplets = use_hard_triplets

    def epoch(self):
        self.model.train()
        for (anchor, positive, negative) in self.dataloader:
            if anchor.size()[0] != self.batch_size:
                continue

            if self.cuda:
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            z_anchor = self.model(anchor)
            z_positive = self.model(positive)

            select_triplets = self.use_hard_triplets
            if self.use_hard_triplets and self.current_epoch < 4:
                negative = negative[:, 0]
                select_triplets = False

            if select_triplets:
                ext_anchor = torch.stack([z_anchor], dim=1)
                n_neg_samples = negative.size(1)
                all_negs = negative.view(-1, negative.size(2), negative.size(3), negative.size(4))
                z_all_negs = self.model(all_negs).view(self.batch_size, n_neg_samples, z_anchor.size(1))
                distances = torch.sum(torch.pow(ext_anchor - z_all_negs, 2), dim=2)
                hard_sample_positions = torch.argmin(distances, dim=1).type(torch.int64)
                batch_indices = torch.arange(0, self.batch_size).type(torch.int64)
                if self.cuda:
                    batch_indices = batch_indices.cuda()
                z_negative = z_all_negs[batch_indices, hard_sample_positions]
            else:
                z_negative = self.model(negative)
            loss = F.triplet_margin_loss(z_anchor, z_positive, z_negative, margin=self.margin)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return {
            "epoch": self.current_epoch,
            "losses": {
                "loss": loss.detach().item(),
            },
            "networks": {
                "model": self.model,
            },
            "optimizers": {
                "optimizer": self.optimizer,
            }
        }