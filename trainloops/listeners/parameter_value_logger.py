"""
    Outputs a per epoch/per model distribution of parameters.
    Parameters can be matched with a name in order to only select certain layers.
"""
import os

import matplotlib.pyplot as plt
import torch
from trainloops.listeners.listener import Listener
import util.output


class ParameterValueLogger(Listener):
    def __init__(self, experiment_output_path, match_string=None):
        super().__init__()
        self.experiment_output_path = experiment_output_path
        name = "param_distributions"
        if match_string is not None:
            name += "_%s"%match_string
        self.path = os.path.join(experiment_output_path, "imgs", name)
        util.output.make_result_dirs(self.path)
        self.match_string = match_string

    def initialize(self):
        pass

    def report(self, state_dict):
        epoch = state_dict["epoch"]

        for m_name, model in state_dict["networks"].items():
            flattened_params = []
            for module in model.children():
                if (self.match_string is None) or (self.match_string in module.__class__.__name__):
                    try:
                        flattened_params.append(module.weight.data.detach().view(-1))
                    except AttributeError:
                        pass
                    try:
                        flattened_params.append(module.bias.data.detach().view(-1))
                    except AttributeError:
                        pass

            if len(flattened_params) == 0:
                continue
            flattened_params = torch.cat(flattened_params, dim=0).cpu().numpy()
            plt.clf()
            plt.xlim((flattened_params.min(), flattened_params.max()))
            plt.hist(flattened_params, bins=40, range=(flattened_params.min(), flattened_params.max()))
            plt.title("Parameters for epoch %d and model %s"%(epoch, m_name))
            plt.savefig(os.path.join(self.path, "%s-epoch-%d.png"%(m_name, epoch)))