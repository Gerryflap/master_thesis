import random

from data.mixture_pairs import MixturePairs


class MixtureTriplets(MixturePairs):

    def __getitem__(self, index):
        x1 = super().__getitem__(index)[0]
        index2 = self.generate_random_same(index)
        x2 = super().__getitem__(index2)[0]
        x3 = super().__getitem__(self.generate_random_different_index(index))[0]
        return x1, x2, x3

    def generate_random_same(self, index):
        block = index//self.datapoints_per_grid_position
        index_in_block = index % self.datapoints_per_grid_position
        index2_in_block = self.generate_random_different_index_generic(index_in_block, self.datapoints_per_grid_position)
        return self.datapoints_per_grid_position * block + index2_in_block