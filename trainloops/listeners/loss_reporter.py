from trainloops.listeners.listener import Listener


class LossReporter(Listener):
    def initialize(self):
        pass

    def report(self, state_dict: dict):
        print("Epoch %d"%state_dict["epoch"])
        for key, value in state_dict.items():
            if "loss" in key:
                print(key, value)
        print()