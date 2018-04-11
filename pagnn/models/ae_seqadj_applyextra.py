import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pagnn
from pagnn.datavargan import dataset_to_datavar
from pagnn.utils import padding_amount, reshape_internal_dim

from .ae_adjacency_conv import AdjacencyConv, AdjacencyConvTranspose
from .ae_sequence_conv import SequenceConv, SequenceConvTranspose
from .ae_sequential import SequentialMod

logger = logging.getLogger(__name__)


class AESeqAdjApplyExtra(nn.Module):

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

        print('kernel size 3 norm')

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
            if i == 0:
                setattr(self, f'encoder_pre_{i}',
                        nn.Sequential(
                            nn.Conv1d(
                                input_channels,
                                output_channels // 2,
                                kernel_size=1,
                                stride=1,
                                padding=0)))
            else:
                setattr(self, f'encoder_pre_{i}', nn.Sequential())
            # Adjacency Conv
            if i < self.n_convs:
                setattr(self, f'encoder_0_{i}',
                        SequentialMod(
                            AdjacencyConv(output_channels // 4, output_channels // 4),
                            nn.LeakyReLU(negative_slope, inplace=True),
                            nn.BatchNorm1d(output_channels // 4),
                        ))
            else:
                setattr(self, f'encoder_0_{i}', SequentialMod())
            # Sequence Conv
            setattr(self, f'encoder_1_{i}',
                    SequentialMod(
                        SequenceConv(output_channels // 2, output_channels, **conv_kwargs),))
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
            if i < (n_layers - 1):
                setattr(self, f'decoder_pre_{i}',
                        nn.Sequential(
                            nn.LeakyReLU(negative_slope, inplace=True),
                            nn.BatchNorm1d(input_channels),
                        ))
            else:
                setattr(self, f'decoder_pre_{i}', nn.Sequential())
            # Sequence Conv
            setattr(self, f'decoder_0_{i}',
                    SequentialMod(
                        SequenceConvTranspose(input_channels, input_channels // 2, **convt_kwargs),
                    ))
            # Adjacency Conv
            if i < self.n_convs:
                setattr(self, f'decoder_1_{i}',
                        SequentialMod(
                            nn.LeakyReLU(negative_slope, inplace=True),
                            nn.BatchNorm1d(input_channels // 4),
                            AdjacencyConvTranspose(getattr(self, f'encoder_0_{i}')[0].spatial_conv),
                        ))
            else:
                setattr(self, f'decoder_1_{i}', SequentialMod())
            # Output
            if i == 0:
                setattr(self, f'decoder_post_{i}',
                        nn.Sequential(
                            nn.Conv1d(
                                input_channels // 2,
                                output_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0),))
            else:
                setattr(self, f'decoder_post_{i}', nn.Sequential())
            input_channels = output_channels

    def forward(self, seq, adjs):
        x = seq

        # Encode
        for i in range(self.n_layers):
            x = getattr(self, f'encoder_pre_{i}')(x)
            x_adj = x[:, x.shape[1] // 2:, :]
            x_adj = getattr(self, f'encoder_0_{i}')(x_adj, i, adjs)
            x = torch.cat([x[:, :x.shape[1] // 2, :], x_adj], 1)
            x = getattr(self, f'encoder_1_{i}')(x, i, adjs)
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
            x = getattr(self, f'decoder_pre_{i}')(x)
            x = getattr(self, f'decoder_0_{i}')(x, i, adjs)
            x_adj = x[:, x.shape[1] // 2:, :]
            x_adj = getattr(self, f'decoder_1_{i}')(x_adj, i, adjs)
            x = torch.cat([x[:, :x.shape[1] // 2, :], x_adj], 1)
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
