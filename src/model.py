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
from pagnn.utils import expand_adjacency_tensor, padding_amount, reshape_internal_dim

logger = logging.getLogger(__name__)


def sparse_sum(input):
    # Need to test if this is faster...
    mone = torch.ones(input.shape[0], 1, dtype=torch.float)
    input = (input @ mone).squeeze()
    return input


class SimpleAdjacencyConv(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, add_counts=True):
        super().__init__()
        # Parameters
        self.normalize = normalize
        self.add_counts = add_counts
        self.takes_extra_args = True
        # Layers
        if add_counts:
            out_channels -= 1
        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor):
        adj_pw = expand_adjacency_tensor(adj)
        adj = adj.to_dense()
        adj_pw = adj_pw.to_dense()
        x = self._conv(x, adj_pw)
        if self.normalize:
            adj_sum = adj.sum(dim=0)
            adj_sum[adj_sum == 0] = 1
            x = x / adj_sum
        if self.add_counts:
            adj_sum = adj.sum(dim=0)
            x = torch.cat([x, adj_sum.expand(x.size(0), 1, -1)], dim=1)
        return x

    def _conv(self, x, adj_pw):
        x = x @ adj_pw.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @ adj_pw[::2, :]
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Conv1d(
            in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False
        )
        if bias:
            self.bias = torch.empty(out_features, dtype=torch.float32, requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.takes_extra_args = True

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.conv.weight.size(1))
        self.conv.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.conv(input)
        output = support @ adj.transpose(0, 1)
        if self.bias is not None:
            output = (output.transpose(1, 2) + self.bias).transpose(2, 1)
        return output


class MaxPoolEverything(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(2)[0]


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

        # === Layer 1 ===
        input_size = self.input_size
        output_size = self.hidden_size

        self.layer_1 = nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, padding=0)

        # === Layer 2 ===
        input_size = output_size
        output_size = int(input_size * 2)

        self.layer_2_seq = nn.Sequential(
            nn.Conv1d(
                input_size // 2,
                output_size // 2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )

        self.layer_2_adj = SequentialMod(
            SimpleAdjacencyConv(input_size // 2, output_size // 2),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        )

        self.layer_2_post = nn.Sequential(nn.ReLU(), nn.Dropout(p=0.5))

        # === Layer N ===
        input_size = output_size
        output_size = 1

        self.layer_n = nn.Sequential(
            MaxPoolEverything(), nn.Linear(input_size, output_size, bias=True)
        )

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
        # Layer 1
        x = self.layer_1(x)
        # Layer 2
        x_seq = x[:, : x.shape[1] // 2, :]
        x_adj = x[:, x.shape[1] // 2 :, :]
        x_seq = self.layer_2_seq(x_seq)
        x_adj = self.layer_2_adj(x_adj, adjs[0][0])
        x = torch.cat([x_seq, x_adj], 1)
        x = self.layer_2_post(x)
        # Layer N
        x = self.layer_n(x)
        x = x.unsqueeze(-1)
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
