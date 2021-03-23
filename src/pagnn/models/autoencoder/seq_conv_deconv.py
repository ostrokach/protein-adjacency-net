import logging
from typing import NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pagnn
from pagnn.utils import padding_amount, reshape_internal_dim

logger = logging.getLogger(__name__)


class SequentialMod(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args_mask = [hasattr(mod, "takes_extra_args") for mod in self._modules.values()]

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
        x = torch.cat([x[:, :, -self.padding :], x, x[:, :, : self.padding]], 2)
        return x.contiguous()


class PermutePadTranspose(nn.Module):
    def __init__(self, padding=2):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        top = x[:, :, : self.padding].clone()
        x[:, :, : self.padding] += x[:, :, -self.padding :]
        x[:, :, -self.padding :] += top
        return x


class CutSequence(nn.Module):
    def __init__(self, offset=1):
        super().__init__()
        self.offset = offset
        self.takes_extra_args = True

    def forward(self, x, seq_len):
        x = x[:, :, self.offset : self.offset + seq_len]
        return x.contiguous()


class AESeqConvDeconv(nn.Module):
    def __init__(
        self,
        n_layers,
        bottleneck_size,
        input_size=20,
        hidden_size=64,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.bottleneck_size = bottleneck_size

        self.n_convs = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        print("0")

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
            setattr(
                self,
                f"encoder_{i}",
                SequentialMod(
                    nn.Conv1d(input_channels, output_channels, **conv_kwargs),
                ),
            )
            if i < (n_layers - 1):
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

        # === Linear ===
        if self.bottleneck_size > 0:
            self.linear_in = nn.Linear(2048, self.bottleneck_size, bias=True)
            self.linear_out = nn.Linear(self.bottleneck_size, 2048, bias=True)
            self.conv_in = nn.Conv1d(
                512, self.bottleneck_size, kernel_size=4, stride=4, padding=0, bias=True
            )
            self.conv_out = nn.Conv1d(
                self.bottleneck_size, 512 * 4, kernel_size=1, stride=1, padding=0, bias=True
            )

        # === Decoder ===
        convt_kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding - 1,
            bias=bias,
        )
        for i in range(n_layers - 1, -1, -1):
            output_channels = input_channels // 2 if i > 0 else input_size
            setattr(
                self,
                f"decoder_{i}",
                SequentialMod(
                    nn.ConvTranspose1d(input_channels, output_channels, **convt_kwargs),
                    CutSequence(0),
                ),
            )
            if i > 0:
                setattr(
                    self,
                    f"decoder_post_{i}",
                    nn.Sequential(
                        nn.ReLU(True),
                        nn.InstanceNorm1d(
                            output_channels, momentum=0.01, affine=True, track_running_stats=True
                        ),
                    ),
                )
            else:
                setattr(self, f"decoder_post_{i}", nn.Sequential())
            input_channels = output_channels

    def forward(self, seq, adjs):
        x = seq

        # Encode
        for i in range(self.n_layers):
            x_list = []
            start = 0
            for adj in adjs:
                seq_len = adj[i].shape[1]
                end = start + seq_len
                assert end <= x.shape[2]
                xd = x[:, :, start:end]
                xd = getattr(self, f"encoder_{i}")(xd, seq_len)
                assert xd.shape[2] == adj[i + 1].shape[1]
                x_list.append(xd)
                start = end
            assert end == x.shape[2]
            x = torch.cat(x_list, 2)
            x = getattr(self, f"encoder_post_{i}")(x)
            logger.debug(f"{i}, {x.shape}")

        # Linear
        #         x_list = []
        #         start = 0
        #         for adj in adjs:
        #             seq_len = adj[i + 1]
        #             end = start + seq_len
        #             assert end <= x.shape[2]
        #             xd = x[:, :, start:end]
        #             pad_amount = padding_amount(xd, 2048)
        #             if pad_amount:
        #                 xd = F.pad(xd, (0, pad_amount))
        #             xd = unfold_to(xd, 2048)
        #             xd = self.linear_in(xd)
        # #             xd = self.conv(xd)
        #             assert np.prod(xd.shape) == 64, xd.shape
        # #             xd = self.convt(xd)
        #             xd = self.linear_out(xd)
        #             xd = unfold_from(xd, n_features)
        #             if pad_amount:
        #                 xd = xd[:, :, :-pad_amount]
        #             x_list.append(xd)
        #             start = end
        #         assert end == x.shape[2]
        #         x = torch.cat(x_list, 2)

        # === Linear ===
        if self.bottleneck_size > 0:

            pad_amount = padding_amount(x, 2048)  # 4 x 512
            if pad_amount:
                x = F.pad(x, (0, pad_amount))

            n_features = x.shape[1]
            x = reshape_internal_dim(x, 1, 512)

            x = self.conv_in(x)
            # x = self.linear_in(x.transpose(1, 2).contiguous())

            assert 0.9 < (np.prod(x.shape) / (seq.shape[2] / 64 * self.bottleneck_size)) <= 1.1, (
                x.shape[1:],
                seq.shape[2] / 64 * self.bottleneck_size,
            )

            # x = self.linear_out(x).transpose(2, 1).contiguous()
            x = self.conv_out(x)

            x = reshape_internal_dim(x, 1, n_features)

            if pad_amount:
                x = x[:, :, :-pad_amount]

        # Decode
        for i in range(self.n_layers - 1, -1, -1):
            x_list = []
            start = 0
            for adj in adjs:
                seq_len = adj[i].shape[1]
                conv_seq_len = adj[i + 1].shape[1]
                end = start + conv_seq_len
                assert end <= x.shape[2]
                xd = x[:, :, start:end]
                xd = getattr(self, f"decoder_{i}")(xd, seq_len)
                x_list.append(xd)
                start = end
            assert end == x.shape[2]
            x = torch.cat(x_list, 2)
            x = getattr(self, f"decoder_post_{i}")(x)
            logger.debug(f"{i}, {x.shape}")

        return x

    def dataset_to_datavar(self, ds):
        # Odd kernel sizes are equivalent to ceil
        # Even kernel sizes are equivalent to floor
        seqs = pagnn.datavargan.push_seqs(ds.seqs)

        class Adj(NamedTuple):
            shape: Tuple[int, int]

        adjs = [
            Adj((0, seqs.shape[2])),
        ]
        for _ in range(self.n_layers + 1):
            length = pagnn.utils.conv1d_shape(adjs[-1].shape[1], kernel_size=3, stride=2, padding=1)
            adj = Adj((0, length))
            adjs.append(adj)

        return pagnn.types.DataVarGAN(seqs, adjs)
