import torch.nn as nn
import torch.nn.functional as F

from .torch_sparse_mm import SparseMM


def adjacency_conv(seq, adj, weight, bias=None):
    """

    Args:
        weight: Size: ``[out_features, in_features, kernel_size]``
    """
    x = seq @ adj.transpose(0, 1)
    x = F.conv1d(x, weight, bias, stride=2, padding=0)
    x = x @ adj[::2, :]
    return x


def adjacency_conv_transpose(seq, adj, weight, bias=None):
    x = seq @ adj.transpose(0, 1)
    # TODO: get rid of detach
    x = F.conv1d(x, weight.detach().transpose(0, 1).contiguous(), bias, stride=2, padding=0)
    x = x @ adj[::2, :]
    return x


class AdjacencyConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)

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
        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.mm2 = SparseMM()

    def forward(self, seq, adj):
        x = self.mm1(seq, adj.transpose(0, 1))
        x = self.spatial_conv(x)
        x = self.mm2(x, adj[::2, :])
        # x = x / adj.sum(dim=0)
        return x
