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
            epochs=1
    ):
        super().__init__(listeners, epochs)
        self.batch_size = dataloader.batch_size
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.cuda = cuda

    def epoch(self):

        for (anchor, positive, negative) in self.dataloader:
            if anchor.size()[0] != self.batch_size:
                continue

            if self.cuda:
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            z_anchor = self.model(anchor)
            z_positive = self.model(positive)
            z_negative = self.model(negative)

            loss = F.triplet_margin_loss(z_anchor, z_positive, z_negative, margin=0.6)

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