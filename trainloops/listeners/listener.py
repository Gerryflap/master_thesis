from abc import ABC, abstractmethod


class Listener(ABC):
    def __init__(self):
        self.trainloop = None

    def register(self, trainloop):
        self.trainloop = trainloop
        self.initialize()

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def report(self, state_dict):
        pass
