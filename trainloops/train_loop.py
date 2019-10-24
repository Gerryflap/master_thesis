"""
    Abstract class for a training loop.
"""
from abc import ABC, abstractmethod


class TrainLoop(ABC):
    def __init__(self, listeners: list, epochs):
        self.listeners = listeners
        self.epochs = epochs
        self.current_epoch = 0
        self.registered = False


    def train(self):
        if not self.registered:
            for listener in self.listeners:
                listener.register(self)
            self.registered = True

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            state_dict = self.epoch()

            for listener in self.listeners:
                listener.report(state_dict)

    @abstractmethod
    def epoch(self):
        """
            Performs one epoch over the dataset and reports the current statistics to the state_dict
            :return: A state dict. This dictionary must contain information required by the listeners
        """
        pass
