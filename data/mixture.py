from torch.utils.data import Dataset
import torch


class MixtureDataset(Dataset):
    def __init__(self, hrange=(-1, 2), vrange=(-1, 2), datapoints_per_grid_position=750, stddev=0.1):
        data = []
        for x in range(hrange[0], hrange[1]+1):
            for y in range(vrange[0], vrange[1]+1):
                data.append(
                    torch.cat([
                        torch.normal(x, stddev, (datapoints_per_grid_position, 1)),
                        torch.normal(y, stddev, (datapoints_per_grid_position, 1))
                    ], dim=1)
                )
        self.data = torch.cat(data, dim=0)

        # Scale the dataset to ensure that it is in the range (-1, 1).
        self.data[:, 0] -= hrange[0]
        self.data[:, 1] -= vrange[0]

        self.data[:, 0] /= (hrange[1] - hrange[0]) * 0.5
        self.data[:, 1] /= (vrange[1] - vrange[0]) * 0.5

        self.data[:, 0] -= 1.0
        self.data[:, 1] -= 1.0

        self.data[:, 0] /= 2.0
        self.data[:, 1] /= 2.0

        self.datapoints_per_grid_position = datapoints_per_grid_position

    def __getitem__(self, item):
        return self.data[item], self.data[item]

    def __len__(self):
        return self.data.size(0)
