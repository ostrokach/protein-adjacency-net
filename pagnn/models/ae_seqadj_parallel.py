import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pagnn
from pagnn.datavargan import dataset_to_datavar
from pagnn.utils import padding_amount, reshape_internal_dim

logger = logging.getLogger(__name__)


class SequentialMod(nn.Sequential):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args_mask = [hasattr(mod, 'takes_extra_args') for mod in self._modules.values()]

    def forward(self, input, *args, **kwargs):
        for module, extra_args in zip(self._modules.values(), self.extra_args_mask):
            if extra_args:
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input


class PermutePad(nn.Module):

    def __init__(self, padding=1):
        super().__init__()
        self.padding = 1

    def forward(self, x):
        x = torch.cat([x[:, :, -self.padding:], x, x[:, :, :self.padding]], 2)
        return x.contiguous()


class PermutePadTranspose(nn.Module):

    def __init__(self, padding=2):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        top = x[:, :, :self.padding].clone()
        x[:, :, :self.padding] += x[:, :, -self.padding:]
        x[:, :, -self.padding:] += top
        return x


class CutSequence(nn.Module):

    def __init__(self, offset=1):
        super().__init__()
        self.offset = offset
        self.takes_extra_args = True

    def forward(self, x, adj):
        x = x[:, :, self.offset:self.offset + adj.shape[1]]
        return x.contiguous()


def get_adjs(seq, n_layers):
    adjs = [seq.shape[2]]
    for _ in range(n_layers):
        adjs.append(pagnn.utils.conv1d_shape(adjs[-1], kernel_size=4, stride=2, padding=1))
    return adjs


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
            # Adjacency conv
            if np.prod(adj[i].shape) == 0:
                xd.zero_()
            else:
                xd = self._conv(xd, adj[i].to_dense())
            # Downsample
            xd = F.avg_pool1d(xd, 2, ceil_mode=True)
            assert xd.shape[2] == adj[i + 1].shape[1]
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
            stop = start + adj[i + 1].shape[1]
            assert stop <= x.shape[2]
            xd = x[:, :, start:stop]
            # Upsample
            #             xd = F.upsample(xd, size=adj[i], mode='linear', align_corners=False)
            xd = F.upsample(xd, scale_factor=2, mode='nearest')
            if (xd.shape[2] - adj[i].shape[1]) == 1:
                xd = xd[:, :, :-1]
            assert xd.shape[2] == adj[i].shape[1]
            assert xd.shape[1] == self.spatial_conv.out_channels
            # Adjacency conv
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


class SequenceConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
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
            bias=bias)
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
                xd = xd[:, :, :adj[i].shape[1]]
            assert xd.shape[2] == adj[i].shape[1]
            x_list.append(xd)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(x_list, 2)
        return x


class AESeqAdjParallel(nn.Module):

    def __init__(self,
                 n_layers,
                 bottleneck_size,
                 input_size=20,
                 hidden_size=64,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 bias=False):
        super().__init__()

        self.n_layers = n_layers
        self.n_convs = 3
        self.bottleneck_size = bottleneck_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        print('---')

        # === Encoder ===
        conv_kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        input_channels = input_size
        for i in range(0, n_layers):
            output_channels = int(input_channels * 2) if i > 0 else hidden_size
            negative_slope = 0.2 if i == 0 else 0.01
            # Input
            setattr(self, f'encoder_pre_{i}',
                    nn.Sequential(
                        nn.Conv1d(
                            input_channels,
                            output_channels // 2,
                            kernel_size=1,
                            stride=1,
                            padding=0),))
            # Downsample
            setattr(self, f'encoder_downsample_seq_{i}',
                    SequentialMod(
                        SequenceConv(output_channels // 4, output_channels // 2, **conv_kwargs),))
            setattr(self, f'encoder_downsample_adj_{i}',
                    SequentialMod(AdjacencyConv(output_channels // 4, output_channels // 2),))
            # Output
            if i < (n_layers - 1):
                setattr(self, f'encoder_post_{i}',
                        nn.Sequential(
                            nn.LeakyReLU(negative_slope, inplace=True),
                            nn.BatchNorm1d(output_channels),
                        ))
            else:
                setattr(self, f'encoder_post_{i}', nn.Sequential())
            input_channels = output_channels

        # === Linear ===
        if self.bottleneck_size > 0:
            self.linear_in = nn.Linear(2048, self.bottleneck_size, bias=True)
            self.linear_out = nn.Linear(self.bottleneck_size, 2048, bias=True)
            self.conv_in = nn.Conv1d(
                512, self.bottleneck_size, kernel_size=4, stride=4, padding=0, bias=True)
            self.conv_out = nn.Conv1d(
                self.bottleneck_size, 512 * 4, kernel_size=1, stride=1, padding=0, bias=True)

        # === Decoder ===
        convt_kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding - 1,
            bias=bias,
        )
        for i in range(n_layers - 1, -1, -1):
            output_channels = input_channels // 2 if i > 0 else input_size
            # Upsample
            setattr(self, f'decoder_upsample_seq_{i}',
                    SequentialMod(
                        SequenceConvTranspose(input_channels // 2, input_channels // 4,
                                              **convt_kwargs),))
            setattr(self, f'decoder_upsample_adj_{i}',
                    SequentialMod(
                        AdjacencyConvTranspose(
                            getattr(self, f'encoder_downsample_adj_{i}')[0].spatial_conv),))
            # Output
            if i > 0:
                setattr(self, f'decoder_post_{i}',
                        nn.Sequential(
                            nn.ReLU(True),
                            nn.BatchNorm1d(input_channels // 2),
                            nn.Conv1d(
                                input_channels // 2,
                                output_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0),
                        ))
            else:
                setattr(
                    self,
                    f'decoder_post_{i}',
                    nn.Sequential(
                        nn.ReLU(True),
                        nn.BatchNorm1d(input_channels // 2),
                        nn.Conv1d(
                            input_channels // 2,
                            output_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                        # nn.Softmax(1),
                    ))
            input_channels = output_channels

    def forward(self, seq, adjs):
        x = seq

        # Encode
        for i in range(self.n_layers):
            x = getattr(self, f'encoder_pre_{i}')(x)
            x_seq = x[:, :x.shape[1] // 2, :]
            x_adj = x[:, x.shape[1] // 2:, :]
            x_seq = getattr(self, f'encoder_downsample_seq_{i}')(x_seq, i, adjs)
            x_adj = getattr(self, f'encoder_downsample_adj_{i}')(x_adj, i, adjs)
            x = torch.cat([x_seq, x_adj], 1)
            x = getattr(self, f'encoder_post_{i}')(x)
            logger.debug(f'{i}, {x.shape}')

        # === Linear ===
        if self.bottleneck_size > 0:

            pad_amount = padding_amount(x, 2048)  # 4 x 512
            if pad_amount:
                x = F.pad(x, (0, pad_amount))

            n_features = x.shape[1]
            x = reshape_internal_dim(x, 1, 512)

            x = self.conv_in(x)
            # x = self.linear_in(x.transpose(1, 2).contiguous())

            assert 0.9 < (np.prod(x.shape) / (seq.shape[2] / 64 * self.bottleneck_size)) <= 1.1, \
                (x.shape[1:], seq.shape[2] / 64 * self.bottleneck_size)

            # x = self.linear_out(x).transpose(2, 1).contiguous()
            x = self.conv_out(x)

            x = reshape_internal_dim(x, 1, n_features)

            if pad_amount:
                x = x[:, :, :-pad_amount]

        # Decode
        for i in range(self.n_layers - 1, -1, -1):
            # x = getattr(self, f'decoder_pre_{i}')(x)
            x_seq = x[:, :x.shape[1] // 2, :]
            x_adj = x[:, x.shape[1] // 2:, :]
            x_seq = getattr(self, f'decoder_upsample_seq_{i}')(x_seq, i, adjs)
            x_adj = getattr(self, f'decoder_upsample_adj_{i}')(x_adj, i, adjs)
            x = torch.cat([x_seq, x_adj], 1)
            x = getattr(self, f'decoder_post_{i}')(x)
            logger.debug(f'{i}, {x.shape}')

        return x

    def get_adjs(self, seq):
        adjs = [seq.shape[2]]
        for _ in range(self.n_layers + 1):
            adjs.append(pagnn.utils.conv1d_shape_ceil(adjs[-1], kernel_size=4, stride=2, padding=1))
        return adjs

    def dataset_to_datavar(self, ds):
        return dataset_to_datavar(
            ds,
            n_convs=self.n_convs + 1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bandwidth=self.kernel_size // 2)


def forward_adj(x, i, adjs, model):
    x_list = []
    start = 0
    for adj in adjs:
        seq_len = adj.shape[1]
        end = start + seq_len
        assert end <= x.shape[2]
        xd = x[:, :, start:end]
        xd = model(xd, adj)
        x_list.append(xd)
        start = end
    assert start == x.shape[2]
    x = torch.cat(x_list, 2)
    return x


def backward_adj_1(x, i, adjs, model):
    x_list = []
    start = 0
    for adj in adjs:
        seq_len = adj[i + 1].shape[1]
        end = start + seq_len
        assert end <= x.shape[2], (end, x.shape, i, adjs)
        xd = x[:, :, start:end]
        xd = model(xd, adj[i])
        x_list.append(xd)
        start = end
    assert start == x.shape[2]
    x = torch.cat(x_list, 2)
    return x


def backward_adj_2(x, i, adjs, model):
    x_list = []
    start = 0
    for adj in adjs:
        seq_len = adj[i].shape[1]
        end = start + seq_len
        assert end <= x.shape[2], (end, x.shape, i, adjs)
        xd = x[:, :, start:end]
        xd = model(xd, adj[i])
        x_list.append(xd)
        start = end
    assert start == x.shape[2]
    x = torch.cat(x_list, 2)
    return x
