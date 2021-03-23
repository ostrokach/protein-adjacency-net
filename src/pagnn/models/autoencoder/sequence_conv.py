import torch
import torch.nn as nn


class SequenceConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.takes_extra_args = True

    def forward(self, x, i, adjs):
        x_list = []
        start = 0
        for adj in adjs:
            stop = start + adj[i].shape[1]
            assert stop <= x.shape[2]
            xd = x[:, :, start:stop]
            xd = self.conv(xd)
            assert xd.shape[2] == adj[i + 1].shape[1]
            x_list.append(xd)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(x_list, 2)
        return x


class SequenceConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.convt = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.takes_extra_args = True

    def forward(self, x, i, adjs):
        x_list = []
        start = 0
        for adj in adjs:
            stop = start + adj[i + 1].shape[1]
            assert stop <= x.shape[2]
            xd = x[:, :, start:stop]
            xd = self.convt(xd)
            if 3 > (xd.shape[2] - adj[i].shape[1]) > 0:
                xd = xd[:, :, : adj[i].shape[1]]
            assert xd.shape[2] == adj[i].shape[1]
            x_list.append(xd)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(x_list, 2)
        return x


class CutSequence(nn.Module):
    def __init__(self, offset=1):
        super().__init__()
        self.offset = offset
        self.takes_extra_args = True

    def forward(self, x, adj):
        x = x[:, :, self.offset : self.offset + adj.shape[1]]
        return x.contiguous()
