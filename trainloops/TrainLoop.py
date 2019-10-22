"""
    Abstract class for a training loop.
"""
from abc import ABC, abstractmethod


class TrainLoop(ABC):
    def __init__(self, listeners: list, epochs=1):
        self.listeners = listeners
        self.epochs = epochs
        self.current_epoch = 0

    def train(self):
        for epoch in range(self.epochs):
            state_dict = self.step()

            for listener in self.listeners:
                listener.report(state_dict)

    @abstractmethod
    def step(self):
        """
            Performs one epoch over the dataset and reports the current statistics to the state_dict
            :return: A state dict. This dictionary must contain information required by the listeners
        """
        pass
