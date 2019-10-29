"""
    A Listener that stores the models and optimizers every n epochs
"""
import os

import torch

import util.output

from trainloops.listeners.listener import Listener


class ModelSaver(Listener):

    def __init__(self, experiment_output_path, n, overwrite=True, print_output=False):
        """
        Initializes the ModelSaver
        :param n: the models and optimizers are stored every n epochs
        :param overwrite: whether to overwrite the saved model every n epochs.
            When False, the epoch will be appended to the filename and a model is stored for every n epochs.

        """
        super().__init__()
        self.experiment_output_path = experiment_output_path
        self.path = os.path.join(experiment_output_path, "params")
        util.output.make_result_dirs(self.path)
        self.n = n
        self.overwrite = overwrite
        self.print_output = print_output

    def initialize(self):
        if self.overwrite:
            util.output.make_result_dirs(os.path.join(self.path, "all_epochs"))

    def report(self, state_dict):

        if state_dict["epoch"] % self.n != 0:
            return

        if not self.overwrite:
            folder_path = os.path.join(self.path, "epoch_%06d" % state_dict["epoch"])
            util.output.make_result_dirs(folder_path)
        else:
            folder_path = os.path.join(self.path, "all_epochs")

        for name, model in state_dict["networks"].items() + state_dict["optimizers"].items():
            fpath = os.path.join(folder_path, name + ".pt")
            with open(fpath, "w") as f:
                torch.save(model, f)

        if self.print_output:
            print("Saved models. ")
