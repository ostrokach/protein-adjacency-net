import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjacencyConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.normalize = True
        self.takes_extra_args = True

    def forward(self, x, i, adjs):
        x_list = []
        start = 0
        for adj in adjs:
            stop = start + adj[i].shape[1]
            assert stop <= x.shape[2]
            xd = x[:, :, start:stop]
            if np.prod(adj[i].shape) == 0:
                xd.zero_()
            else:
                xd = self._conv(xd, adj[i].to_dense())
            x_list.append(xd)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(x_list, 2)
        return x

    def _conv(self, x, adj):
        x = x @ adj.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @ adj[::2, :]
        if self.normalize:
            adj_sum = adj.sum(dim=0)
            adj_sum[adj_sum == 0] = 1
            x = x / adj_sum
        return x


class AdjacencyConvTranspose(nn.Module):

    def __init__(self, spatial_conv):
        super().__init__()
        self.spatial_conv = spatial_conv
        self.normalize = True
        self.takes_extra_args = True

    def forward(self, x, i, adjs):
        x_list = []
        start = 0
        for adj in adjs:
            stop = start + adj[i].shape[1]
            assert stop <= x.shape[2]
            xd = x[:, :, start:stop]
            if np.prod(adj[i].shape) == 0:
                xd.zero_()
            else:
                xd = self._conv(xd, adj[i].to_dense())
            x_list.append(xd)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(x_list, 2)
        return x

    def _conv(self, x, adj):
        x = x @ adj.transpose(0, 1)
        x = F.conv1d(x, self.spatial_conv.weight.transpose(0, 1).contiguous(), stride=2, padding=0)
        x = x @ adj[::2, :]
        if self.normalize:
            adj_sum = adj.sum(dim=0)
            adj_sum[adj_sum == 0] = 1
            x = x / adj_sum
        return x
