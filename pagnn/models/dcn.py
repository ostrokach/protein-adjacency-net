import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from pagnn.datavargan import dataset_to_datavar
from pagnn.utils import padding_amount, reshape_internal_dim

from .common import AdjacencyConv, SequenceConv, SequentialMod

logger = logging.getLogger(__name__)


class DCN(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_convs: int = 3,
        input_size: int = 20,
        hidden_size: int = 64,
        bottleneck_size: int = 16,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_convs = n_convs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self._configure_encoder()

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
                setattr(
                    self,
                    f"encoder_pre_{i}",
                    nn.Sequential(
                        nn.Conv1d(
                            input_channels, output_channels // 2, kernel_size=1, stride=1, padding=0
                        )
                    ),
                )
            else:
                setattr(self, f"encoder_pre_{i}", nn.Sequential())
            # Adjacency Conv
            if i < self.n_convs:
                setattr(
                    self,
                    f"encoder_0_{i}",
                    SequentialMod(
                        AdjacencyConv(output_channels // 4, output_channels // 4),
                        nn.LeakyReLU(negative_slope, inplace=True),
                        nn.InstanceNorm1d(
                            output_channels // 4,
                            momentum=0.01,
                            affine=True,
                            track_running_stats=True,
                        ),
                    ),
                )
            else:
                setattr(self, f"encoder_0_{i}", SequentialMod())
            # Sequence Conv
            setattr(
                self,
                f"encoder_1_{i}",
                SequentialMod(SequenceConv(output_channels // 2, output_channels, **conv_kwargs)),
            )
            if i < (self.n_layers - 1):
                setattr(
                    self,
                    f"encoder_post_{i}",
                    nn.Sequential(
                        nn.LeakyReLU(negative_slope, inplace=True),
                        nn.InstanceNorm1d(
                            output_channels, momentum=0.01, affine=True, track_running_stats=True
                        ),
                    ),
                )
            else:
                setattr(self, f"encoder_post_{i}", nn.Sequential())
            input_channels = output_channels

        if self.bottleneck_size > 0:
            self.linear_in = nn.Linear(2048, self.bottleneck_size, bias=True)
            self.conv_in = nn.Conv1d(
                512, self.bottleneck_size, kernel_size=4, stride=4, padding=0, bias=True
            )

        return input_channels

    def forward(self, seq, adjs):
        for i in range(self.n_layers):
            x = getattr(self, f"encoder_pre_{i}")(x)
            x_adj = x[:, x.shape[1] // 2 :, :]
            x_adj = getattr(self, f"encoder_0_{i}")(x_adj, i, adjs)
            x = torch.cat([x[:, : x.shape[1] // 2, :], x_adj], 1)
            x = getattr(self, f"encoder_1_{i}")(x, i, adjs)
            x = getattr(self, f"encoder_post_{i}")(x)
            # logger.debug(f"Encoder layer: {i}, input shape: {x.shape}")

        if self.bottleneck_size > 0:
            pad_amount = padding_amount(x, 2048)  # 4 x 512
            if pad_amount:
                x = F.pad(x, (0, pad_amount))
            x = reshape_internal_dim(x, 1, 512)
            x = self.conv_in(x)
            # x = self.linear_in(x.transpose(1, 2).contiguous())

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
