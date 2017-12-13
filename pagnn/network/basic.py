"""Basic single hidden layer neural networks."""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        n_filters = 12
        self.spatial_conv = nn.Conv1d(20, n_filters, 2, stride=2, bias=False)
        self.combine_convs = nn.Linear(n_filters, 1, bias=False)

    def forward(self, aa, adjacency):
        x = aa @adjacency.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @adjacency[::2, :]
        # import pdb; pdb.set_trace()
        # x = x.sum(dim=0) / adjacency.sum(dim=0)
        x, idxs = (x / adjacency.sum(dim=0)).max(dim=2)
        x = self.combine_convs(x)
        x = x.squeeze()
        return F.sigmoid(x)
