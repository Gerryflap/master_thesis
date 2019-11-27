from torch.utils.data import DataLoader

from trainloops.listeners.listener import Listener


class DiscriminatorOverfitMonitor(Listener):
    def __init__(self, train_dataset, validation_dataset, n_images, args):
        super().__init__()
        self.n_images = n_images
        self.cuda = args.cuda
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.tloader = DataLoader(validation_dataset, self.n_images, True)
        self.vloader = DataLoader(validation_dataset, self.n_images, True)

    def initialize(self):
        self.x_train = self.tloader.__iter__().__next__()[0]
        if self.cuda:
            self.x_train = self.x_train.cuda()

        # Remove the loader since we got our test images
        del self.tloader

        self.x_valid = self.vloader.__iter__().__next__()[0]
        if self.cuda:
            self.x_valid = self.x_valid.cuda()

        # Remove the loader since we got our test images
        del self.vloader

    def report(self, state_dict):
        if "D" in state_dict["networks"]:
            D = state_dict["networks"]["D"]
        else:
            raise ValueError("Could not find a discriminator network in the state dict!")

        d_train = D(self.x_train).mean().detach().cpu().item()
        d_valid = D(self.x_valid).mean().detach().cpu().item()

        print("D(train) mean: ", d_train)
        print("D(valid) mean: ", d_valid)
        print()