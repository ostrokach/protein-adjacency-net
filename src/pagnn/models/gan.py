"""

Sources:

- https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
- https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/\
deep_convolutional_gan/model.py
"""
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pagnn.models import AdjacencyConv, adjacency_conv_transpose
from pagnn.utils import conv1d_shape


def conv1d_xtimes_shape(in_channels, n_times, kernel_size=1, stride=1, padding=0, dilation=1):
    out_channels = in_channels
    for _ in range(n_times):
        out_channels = conv1d_shape(
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    return out_channels


class GANParams:
    kernel_size: int = 4
    stride: int = 2
    padding: int = 0

    x_in: int = 20
    x_hidden: int = 32
    x_out: int = 64
    z_in: int = 50
    z_out: int = 2048
    n_layers: int = 7
    aa_per_prediction: int = x_out


class DiscriminatorNet(nn.Module, GANParams):
    def __init__(self):
        # isize=512, nz=100, nc=20, ndf=32
        """
        Args:
            isize: Image size (must be a power of two; defaults to 256).
            nz: Dimensionality of latent variable ``z``.
            nc: Dimensionality of input variable ``x``.
            ndf:

        Notes:
            Use non-overlapping stride to avoid checkboard artifacts:
                https://distill.pub/2016/deconv-checkerboard/
        """
        super().__init__()

        model = OrderedDict()

        # 20 x 64+
        for i in range(1):
            model[f"adjacency_conv_{i}"] = AdjacencyConv(self.x_in, self.x_hidden)
            model[f"conv_{i}"] = nn.Conv1d(
                self.x_hidden,
                self.x_out,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
            model[f"leaky_relu_{i}"] = nn.LeakyReLU(0.2)
            in_feat = self.x_out

        for i in range(1, 4):
            out_feat = in_feat * 2
            model[f"adjacency_conv_{i}"] = AdjacencyConv(in_feat, in_feat)
            model[f"conv_{i}"] = nn.Conv1d(
                in_feat,
                out_feat,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
            model[f"instance_norm_{i}"] = nn.InstanceNorm1d(out_feat, affine=True)
            model[f"leaky_relu_{i}"] = nn.LeakyReLU(0.2)
            in_feat = out_feat

        for i in range(4, 6):
            out_feat = in_feat * 2
            model[f"conv_{i}"] = nn.Conv1d(
                in_feat,
                out_feat,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
            model[f"instance_norm_{i}"] = nn.InstanceNorm1d(out_feat, affine=True)
            model[f"leaky_relu_{i}"] = nn.LeakyReLU(0.2)
            in_feat = out_feat

        assert in_feat == self.z_out, in_feat

        for i in range(6, 7):
            out_feat = 1
            model[f"conv_{i}"] = nn.Conv1d(
                in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False
            )
            model[f"sigmoid_{i}"] = nn.Sigmoid()

        assert (i + 1) == self.n_layers

        for key, value in model.items():
            setattr(self, key, value)
        self.model = model

    def forward(self, seq: Variable, adjs: List[List[Variable]]):
        """
        Args:
            seq: A sequence of ``128 x N`` amino acids (if your sequence is shorter, pad it with
                zeros). Shape: ``[?, 20, 128 x N]``.
            adjs:
                Shape: ``[ [xxx, 512], [xxx, 256], [xxx, 128], [xxx, 64] ]``.
        """
        x = seq

        # 20 x 512
        for i in range(0, 6):
            # Adjacency convolutions
            if i in range(0, 4):
                start = 0
                num_odd = 0
                seq_slices = []
                for j, adj in enumerate(adjs):
                    stop = start + adj[i].shape[1]
                    assert stop <= x.shape[2], (i, start, stop, x.shape, len(adjs))
                    seq_slice = x[:, :, start:stop]
                    seq_slice = self.model[f"adjacency_conv_{i}"](seq_slice, adj[i])
                    seq_slices.append(seq_slice)
                    if i == 0 or adj[i - 1].shape[1] % 2 == 0:
                        start = stop
                    else:
                        num_odd += 1
                        if num_odd % 2 == 0:
                            start = stop + 1
                        else:
                            start = stop
                assert (x.shape[2] - stop) <= 1, (i, start, stop, x.shape, len(adjs))
                x = torch.cat(seq_slices, 2)
            # Regular convolutions
            x = torch.cat([x[:, :, -1:], x, x[:, :, :1]])
            x = self.model[f"conv_{i}"](x)
            # Batch normalization
            if i in range(1, 999):
                x = self.model[f"instance_norm_{i}"](x)
            # Non-linearity
            x = self.model[f"leaky_relu_{i}"](x)

        for i in range(6, 7):
            x = self.model[f"conv_{i}"](x)
            x = self.model[f"sigmoid_{i}"](x)

        return x.view(-1, 1)


class GeneratorNet(nn.Module, GANParams):
    def __init__(self):
        """
        Args:
            isize: Sequence length (at present, this has to be 512).
            nz: Size of the latent z vector.
            nc: Input sequence channels (20).
            ngf:

        Notes:
            Use non-overlapping stride to avoid checkboard artifacts:
                https://distill.pub/2016/deconv-checkerboard/
        """
        super().__init__()

        model = OrderedDict()

        assert self.n_layers - 1 == 6

        # ? x 1
        for i in range(6, 5, -1):
            model[f"linear_{i}"] = nn.Linear(self.z_in, self.z_out, bias=False)
            model[f"instance_norm_{i}"] = nn.InstanceNorm1d(self.z_out, affine=True)
            model[f"relu_{i}"] = nn.ReLU()
            in_feat = self.z_out

        # 2048 x 4
        for i in range(5, 0, -1):
            out_feat = in_feat // 2
            model[f"convt_{i}"] = nn.ConvTranspose1d(
                in_feat,
                out_feat,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
            model[f"instance_norm_{i}"] = nn.InstanceNorm1d(out_feat, affine=True)
            model[f"relu_{i}"] = nn.ReLU()
            in_feat = out_feat

        assert in_feat == self.x_out, in_feat

        for i in range(0, -1, -1):
            model[f"convt_{i}"] = nn.ConvTranspose1d(
                self.x_out,
                self.x_hidden,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )

        assert i == 0

        for key, value in model.items():
            setattr(self, key, value)
        self.model = model

    def forward(self, z: Variable, adjs: List[Variable], net_d):
        i = 6
        x = self._expand_z(z, adjs, self.model[f"linear_{i}"])
        x = self.model[f"relu_{i}"](x)

        # 2048 x 4
        for i in range(5, -1, -1):
            print("gen i:", i)
            # Regular convolutions
            x = self.model[f"convt_{i}"](x)
            # Adjacency convolutions
            if i in range(0, 4):
                x = self._adjacency_conv(i, x, net_d, adjs)
            # Batch normalization
            if i in range(1, 999):
                x = self.model[f"instance_norm_{i}"](x)
            # Non-linearity
            if i in range(1, 999):
                x = self.model[f"relu_{i}"](x)

        return F.softmax(x, 1)

    def _get_num_preds(self, adjs):
        input_size = sum(adj[0].shape[1] for adj in adjs)
        num_preds = conv1d_xtimes_shape(
            input_size,
            self.n_layers - 1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
        )
        return num_preds

    def _expand_z(self, z, adjs, linear):
        x_list = []
        start = 0
        # total_length = sum(adj[0].shape[1] for adj in adjs)
        print("Num preds:", self._get_num_preds(adjs))
        for _ in range(self._get_num_preds(adjs)):
            stop = start + self.z_in
            assert stop <= z.shape[1], (start, stop, z.shape)
            x = linear(z[:, start:stop])
            x_list.append(x)
            start = stop
        x = torch.stack(x_list, 2)
        return x

    def _adjacency_conv(self, i, x, net_d, adjs):
        adjacency_conv_weight = getattr(net_d, f"adjacency_conv_{i}").spatial_conv.weight

        start = 0
        num_odd = 0
        seq_list = []
        for j, adj in enumerate(adjs):
            stop = start + adj[i].shape[1]
            assert stop <= x.shape[2], (i, start, stop, x.shape, len(adjs))
            seq_slice = x[:, :, start:stop]
            seq_slice = adjacency_conv_transpose(seq_slice, adj[i], adjacency_conv_weight)
            if i == 0 or adj[i - 1].shape[1] % 2 == 0:
                start = stop
            else:
                num_odd += 1
                if num_odd % 2 == 0:
                    start = stop + 1
                else:
                    start = stop
            seq_list.append(seq_slice)
        # assert (x.shape[2] - stop) <= 1, (i, start, stop, x.shape, len(adjs))
        print("x.shape[2]:", x.shape[2])
        print("stop:", stop)
        x = torch.cat(seq_list, 2)
        return x
