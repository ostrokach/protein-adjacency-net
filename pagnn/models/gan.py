"""

Sources:

- https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
- https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/\
deep_convolutional_gan/model.py
"""
import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .torch_adjacency_conv import AdjacencyConv, adjacency_conv_transpose


class GANParams:
    x_in: int = 20
    x_hidden: int = 32
    x_out: int = 64
    z_in: int = 50
    z_out: int = 2048
    n_layers: int = 7
    aa_per_prediction: int = 64


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
            model[f'adjacency_conv_{i}'] = AdjacencyConv(self.x_in, self.x_hidden)
            model[f'conv_{i}'] = nn.Conv1d(
                self.x_hidden, self.x_out, kernel_size=4, stride=2, padding=0, bias=False)
            model[f'leaky_relu_{i}'] = nn.LeakyReLU(0.2)
            in_feat = self.x_out

        for i in range(1, 4):
            out_feat = in_feat * 2
            model[f'adjacency_conv_{i}'] = AdjacencyConv(in_feat, in_feat)
            model[f'conv_{i}'] = nn.Conv1d(
                in_feat, out_feat, kernel_size=4, stride=2, padding=0, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'leaky_relu_{i}'] = nn.LeakyReLU(0.2)
            in_feat = out_feat

        for i in range(4, 6):
            out_feat = in_feat * 2
            model[f'conv_{i}'] = nn.Conv1d(
                in_feat, out_feat, kernel_size=4, stride=2, padding=0, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'leaky_relu_{i}'] = nn.LeakyReLU(0.2)
            in_feat = out_feat

        assert in_feat == self.z_out, in_feat

        for i in range(6, 7):
            out_feat = 1
            model[f'conv_{i}'] = nn.Conv1d(
                in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False)
            model[f'sigmoid_{i}'] = nn.Sigmoid()

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
                seq_slices = []
                start = 0
                prev_idx_bool = True
                for j, adj in enumerate(adjs):
                    stop = start + adj[i].shape[1]
                    if stop > x.shape[2]:
                        # import pdb; pdb.set_trace()
                        pass
                    assert stop <= x.shape[2], (i, start, stop, x.shape, len(adjs))
                    seq_slice = x[:, :, start:stop]
                    seq_slice = self.model[f'adjacency_conv_{i}'](seq_slice, adj[i])
                    seq_slices.append(seq_slice)
                    start = stop
                assert 0 <= (x.shape[2] - stop) < 2, (i, start, stop, x.shape, len(adjs))
                x = torch.cat(seq_slices, 2)
            # Regular convolutions
            x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], 2)
            if i in range(0, 4):
                seq_slices = []
                start = 0
                for j, adj in enumerate(adjs):
                    stop = start + adj[i].shape[1]
                    if stop > x.shape[2]:
                        # import pdb; pdb.set_trace()
                        pass
                    assert stop <= x.shape[2], (i, start, stop, x.shape, len(adjs))
                    seq_slice = x[:, :, start:stop + 2]
                    seq_slice = self.model[f'adjacency_conv_{i}'](seq_slice, adj[i])
                    seq_slices.append(seq_slice)
                    if i == 0:
                        start = stop
                    elif adj[i - 1].shape[1] % 2 == 0:
                        start = stop
                    else:
                        if prev_idx_bool:
                            start = stop - 1
                        else:
                            start = stop
                        prev_idx_bool = not prev_idx_bool
                assert 0 <= (x.shape[2] - stop) < 2, (i, start, stop, x.shape, len(adjs))
                x = torch.cat(seq_slices, 2)
            # x = self.model[f'conv_{i}'](x)
            if i in range(1, 6):
                x = self.model[f'instance_norm_{i}'](x)
            x = self.model[f'leaky_relu_{i}'](x)

        for i in range(6, 7):
            x = self.model[f'conv_{i}'](x)
            x = self.model[f'sigmoid_{i}'](x)

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

        # ? x 1
        for i in range(1):
            model[f'linear_{i}'] = nn.Linear(self.z_in, self.z_out, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(self.z_out, affine=False)
            model[f'relu_{i}'] = nn.ReLU()
            in_feat = self.z_out

        # 2048 x 4
        for i in range(1, 6):
            out_feat = in_feat // 2
            model[f'convt_{i}'] = nn.ConvTranspose1d(
                in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'relu_{i}'] = nn.ReLU()
            in_feat = out_feat

        assert in_feat == self.x_out, in_feat

        for i in range(6, 7):
            model[f'convt_{i}'] = nn.ConvTranspose1d(
                self.x_out, self.x_hidden, kernel_size=4, stride=2, padding=1, bias=False)

        assert (i + 1) == self.n_layers

        for key, value in model.items():
            setattr(self, key, value)
        self.model = model

    def forward(self, z: Variable, adjs: List[Variable], net_d):
        i = 0
        x = self._expand_z(z, adjs, self.model[f'linear_{i}'])
        x = self.model[f'relu_{i}'](x)

        # 2048 x 4
        for i in range(1, 7):
            x = self.model[f'convt_{i}'](x)
            if i in range(3, 999):  # 3, 4, 5, 6
                x = self._adjacency_conv(x, adjs, net_d, i)
            if i in range(0, 6):  # all but last
                x = self.model[f'instance_norm_{i}'](x)
            if i in range(0, 6):  # all but last
                x = self.model[f'relu_{i}'](x)

        return F.softmax(x, 1)

    def _expand_z(self, z, adjs, linear):
        x_list = []
        start = 0
        total_length = sum(adj[0].shape[1] for adj in adjs)
        for _ in range(math.ceil(total_length / self.x_out)):
            # import pdb; pdb.set_trace()
            stop = start + self.z_in
            x_list.append(linear(z[:, start:stop]))
            start = stop
        x = torch.stack(x_list, 2)
        return x

    def _adjacency_conv(self, x, adjs, net_d, i):
        adj_idx = self.n_layers - 1 - i
        adjacency_conv_weight = getattr(net_d, f'adjacency_conv_{adj_idx}').spatial_conv.weight

        start = 0
        seq_list = []
        for adj in adjs:
            stop = start + adj.shape[1]
            seq = x[start:stop]
            seq = adjacency_conv_transpose(seq, adj[adj_idx], adjacency_conv_weight)
            seq_list.append(seq)
            start = stop
        assert start == x.shape[2]
        x = torch.cat(seq_list, 2)
        return x
