import torch.nn as nn

from .sparse_mm import SparseMM


class AdjacencyConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Conv1d(in_channels, out_channels, 2, 2, bias=False)

    def forward(self, seq, adj):
        x = seq @ adj.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @ adj[::2, :]
        # x = x / adj.sum(dim=0)
        return x


class AdjacencyConvSparse(nn.Module):

    def __init__(self, in_channels, out_channels, normalize=False):
        super().__init__()
        self.mm1 = SparseMM()
        self.spatial_conv = nn.Conv1d(in_channels, out_channels, 2, 2, bias=False)
        self.mm2 = SparseMM()

    def forward(self, seq, adj):
        x = self.mm1(seq, adj.transpose(0, 1))
        x = self.spatial_conv(x)
        x = self.mm2(x, adj[::2, :])
        # x = x / adj.sum(dim=0)
        return x
