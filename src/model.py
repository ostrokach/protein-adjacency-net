"""
This is a basic ``seq+adj - conv - seq+adj`` network.

"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pagnn.models.dcn
from pagnn.datavargan import dataset_to_datavar
from pagnn.models.common import AdjacencyConv, SequenceConv, SequentialMod
from pagnn.utils import padding_amount, reshape_internal_dim

logger = logging.getLogger(__name__)


class SimpleAdjacencyConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.normalize = True
        self.takes_extra_args = True

    def forward(self, x, adj):
        if adj.layout == torch.sparse_coo:
            adj = adj.to_dense()
        x = self._conv(x, adj)
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


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.FloatTensor(in_features, out_features, requires_grad=True)
        if bias:
            self.bias = torch.FloatTensor(out_features, requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Custom(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        n_convs: int = 3,
        input_size: int = 20,
        hidden_size: int = 64,
        bottleneck_size: int = 0,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        logger.info("Network name: '%s'", self.__class__.__name__)

        self.n_layers = n_layers
        self.n_convs = n_convs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # self._configure_encoder()
        self.encoder = SequentialMod(
            SimpleAdjacencyConv(self.input_size, 64), nn.ReLU(inplace=True)
        )
        self.linear_in = nn.Linear(64, 1, bias=True)

    def _configure_encoder(self):
        conv_kwargs = dict(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias
        )

        input_channels = self.input_size
        for i in range(0, self.n_layers):
            output_channels = int(input_channels * 2) if i > 0 else self.hidden_size
            negative_slope = 0.2 if i == 0 else 0.01

            # Input
            if i == 0:
                encoder_pre = nn.Sequential(
                    nn.Conv1d(
                        input_channels, output_channels // 2, kernel_size=1, stride=1, padding=0
                    )
                )
            elif i % 2 == 1:
                encoder_pre = nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, **conv_kwargs)
                )
            else:
                encoder_pre = nn.Sequential()
            setattr(self, f"encoder_pre_{i}", encoder_pre)

            # Sequence Conv
            if i % 2 == 0:
                encoder_seq = SequentialMod(
                    SequenceConv(output_channels // 4, output_channels // 2, **conv_kwargs)
                )
            else:
                encoder_seq = SequentialMod()
            setattr(self, f"encoder_seq_{i}", encoder_seq)

            # Adjacency Conv
            if i % 2 == 0:
                encoder_adj = SequentialMod(
                    AdjacencyConv(output_channels // 4, output_channels // 2),
                    nn.Conv1d(output_channels // 2, output_channels // 2, **conv_kwargs),
                    # nn.LeakyReLU(negative_slope, inplace=True),
                    # nn.InstanceNorm1d(
                    #     output_channels // 4,
                    #     momentum=0.01,
                    #     affine=True,
                    #     track_running_stats=True,
                    # ),
                )
            else:
                encoder_adj = SequentialMod()
            setattr(self, f"encoder_adj_{i}", encoder_adj)

            # Output
            if i < (self.n_layers - 1):
                encoder_post = nn.Sequential(
                    nn.LeakyReLU(negative_slope, inplace=True),
                    nn.BatchNorm1d(
                        output_channels, momentum=0.01, affine=True, track_running_stats=True
                    ),
                )
            else:
                encoder_post = nn.Sequential()
            setattr(self, f"encoder_post_{i}", encoder_post)

            input_channels = output_channels

        logger.info("Final output_channels: %s", output_channels)

        if self.bottleneck_size == 0:
            self.linear_in = nn.Linear(output_channels, 1, bias=True)
            self.conv_in = nn.Conv1d(output_channels, 1, kernel_size=output_channels, bias=True)
        else:
            raise NotImplementedError
            self.linear_in = nn.Linear(2048, self.bottleneck_size, bias=True)
            self.conv_in = nn.Conv1d(
                512, self.bottleneck_size, kernel_size=4, stride=4, padding=0, bias=True
            )

        return input_channels

    def forward(self, seq, adjs):
        x = seq
        x = self.encoder(x, adjs[0][0])
        x = x.max(2)[0]
        x = self.linear_in(x).unsqueeze(-1)
        return x

    def forward_bak(self, seq, adjs):
        x = seq

        for i in range(self.n_layers):
            x = getattr(self, f"encoder_pre_{i}")(x)
            x_seq = x[:, : x.shape[1] // 2, :]
            x_seq = getattr(self, f"encoder_seq_{i}")(x_seq, i, adjs)
            x_adj = x[:, x.shape[1] // 2 :, :]
            x_adj = getattr(self, f"encoder_adj_{i}")(x_adj, i, adjs)
            x = torch.cat([x_seq, x_adj], 1)
            x = getattr(self, f"encoder_post_{i}")(x)
            # logger.debug(f"Encoder layer: {i}, input shape: {x.shape}")

        if self.bottleneck_size == 0:
            x = x.max(2, keepdim=True)[0]
            x = self.linear_in(x.squeeze()).unsqueeze(-1)
        else:
            raise NotImplementedError
            pad_amount = padding_amount(x, 2048)  # 4 x 512
            if pad_amount:
                x = F.pad(x, (0, pad_amount))
            x = reshape_internal_dim(x, 1, 512)
            x = self.conv_in(x)

        return x

    def dataset_to_datavar(self, ds):
        return dataset_to_datavar(
            ds,
            n_convs=self.n_convs + 1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            remove_diags=1 + self.kernel_size // 2,
            add_diags=0,
        )


pagnn.models.dcn.Custom = Custom
