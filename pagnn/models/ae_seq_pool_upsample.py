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

    def forward(self, x, seq_len):
        x = x[:, :, self.offset:self.offset + seq_len]
        return x.contiguous()


class Downsample(nn.Module):

    def forward(self, x, i, adjs):
        x_list = []
        start = 0
        for adj in adjs:
            stop = start + adj[i].shape[1]
            assert stop <= x.shape[2]
            xd = x[:, :, start:stop]
            xd = F.avg_pool1d(xd, 2, ceil_mode=True)
            assert xd.shape[2] == adj[i + 1].shape[1]
            x_list.append(xd)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(x_list, 2)
        return x


class Upsample(nn.Module):

    def forward(self, x, i, adjs):
        x_list = []
        start = 0
        for adj in adjs:
            stop = start + adj[i + 1].shape[1]
            assert stop <= x.shape[2]
            xd = x[:, :, start:stop]
            #             xd = F.upsample(xd, size=adj[i], mode='linear', align_corners=False)
            xd = F.upsample(xd, scale_factor=2, mode='nearest')
            if (xd.shape[2] - adj[i].shape[1]) == 1:
                xd = xd[:, :, :-1]
            assert xd.shape[2] == adj[i].shape[1]
            x_list.append(xd)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(x_list, 2)
        return x


class AESeqPoolUpsample(nn.Module):

    def __init__(self,
                 n_layers,
                 bottleneck_size,
                 input_size=20,
                 hidden_size=64,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 bias=False):
        super().__init__()

        self.n_layers = n_layers
        self.bottleneck_size = bottleneck_size

        self.n_convs = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # === Encoder ===
        conv_kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        # === Encoder ===
        input_channels = input_size
        for i in range(n_layers):
            output_channels = int(input_channels * 2) if i > 0 else hidden_size
            negative_slope = 0.01 if i > 0 else 0.2
            if i < (n_layers - 1):
                setattr(self, f'encoder_pre_{i}',
                        nn.Sequential(
                            nn.Conv1d(input_channels, output_channels, **conv_kwargs),
                            nn.LeakyReLU(negative_slope, inplace=True),
                            nn.BatchNorm1d(output_channels),
                        ))
            else:
                setattr(self, f'encoder_pre_{i}',
                        nn.Sequential(nn.Conv1d(input_channels, output_channels, **conv_kwargs),))
            setattr(self, f'encoder_downsample_{i}', Downsample())
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
        for i in range(n_layers - 1, 0, -1):
            output_channels = input_channels // 2
            setattr(self, f'decoder_pre_{i}', nn.Sequential())
            setattr(self, f'decoder_upsample_{i}', Upsample())
            setattr(self, f'decoder_post_{i}',
                    nn.Sequential(
                        nn.Conv1d(input_channels, output_channels, **conv_kwargs),
                        nn.ReLU(True),
                        nn.BatchNorm1d(output_channels),
                    ))
            input_channels = output_channels
        setattr(self, 'decoder_pre_0', nn.Sequential())
        setattr(self, 'decoder_upsample_0', Upsample())
        setattr(self, 'decoder_post_0',
                nn.Sequential(nn.Conv1d(input_channels, input_size, **conv_kwargs),))

    def forward(self, seq, adjs):
        x = seq

        # Encode
        for i in range(self.n_layers):
            x = getattr(self, f'encoder_pre_{i}')(x)
            x = getattr(self, f'encoder_downsample_{i}')(x, i, adjs)
            x = getattr(self, f'encoder_post_{i}')(x)

        # Linear
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
            x = getattr(self, f'decoder_upsample_{i}')(x, i, adjs)
            x = getattr(self, f'decoder_post_{i}')(x)

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
