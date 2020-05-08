import json
import os
from collections import defaultdict

from trainloops.listeners.listener import Listener
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class LossPlotter(Listener):
    def __init__(self, output_path, every_n_epochs=1):
        super().__init__()
        self.path = os.path.join(output_path, "imgs")
        self.every_n_epochs = every_n_epochs
        self.losses = defaultdict(lambda: [])

    def initialize(self):
        pass

    def report(self, state_dict: dict):
        epoch = state_dict["epoch"]
        if epoch % self.every_n_epochs != 0.0:
            return

        for key, value in state_dict["losses"].items():
            self.losses[key].append(value)

        plt.clf()
        for loss_name, loss_progression in self.losses.items():
            plt.plot(loss_progression, label=loss_name)
        plt.legend()
        plt.savefig(os.path.join(self.path, "loss_plot.png"))

        with open(os.path.join(self.path, "loss_progression.json"), "w") as f:
            json.dump(self.losses, f)
