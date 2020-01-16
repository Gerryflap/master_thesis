import random

from data.mixture import MixtureDataset


class MixturePairs(MixtureDataset):

    def __getitem__(self, index):
        x1 = super().__getitem__(index)[0]
        x2 = super().__getitem__(self.generate_random_different_index(index))[0]
        return x1, x2

    def generate_random_different_index(self, index):
        # Generate a random number between 0 and len(ident_list) - 2
        # (the minus 2 is because 1. randint includes the end value and 2. we do not want to include index.
        rand_index = random.randint(0, len(self)-2)

        # Skip the current index, because we don't want it to be included
        if rand_index >= index:
            rand_index += 1

        return rand_index