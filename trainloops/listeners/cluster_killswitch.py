"""
    Cancelling jobs on the University cluster forces programs to instantly quit,
        which sometimes crashes cluster nodes.
    As a remedy, this killswitch listener will stop the experiment in a nicer way to prevent this from happening.
    The experiment will be stopped if a file named "stop" is encountered in the results folder of the experiment.
        The existence of this file is checked after each epoch.
"""
import os

from trainloops.listeners.listener import Listener


class KillSwitchListener(Listener):
    def __init__(self, experiment_path):
        super().__init__()
        self.path = os.path.join(experiment_path, "stop")

    def initialize(self):
        pass

    def report(self, state_dict):
        if os.path.exists(self.path):
            exit()
