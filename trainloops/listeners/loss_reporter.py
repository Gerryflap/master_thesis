from trainloops.listeners.listener import Listener


class LossReporter(Listener):
    def __init__(self, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def initialize(self):
        pass

    def report(self, state_dict: dict):
        if state_dict["epoch"]%self.every_n_epochs != 0.0:
            return

        print("Epoch %d"%state_dict["epoch"])
        for key, value in state_dict["losses"].items():
            print(key + ": ", value)
        print()
