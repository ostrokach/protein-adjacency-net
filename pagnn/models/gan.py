"""

Sources:

- https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
- https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/\
deep_convolutional_gan/model.py
"""
from collections import OrderedDict
from typing import List

import torch.nn as nn
from torch.autograd import Variable

from .torch_adjacency_conv import AdjacencyConv


class Discriminator(nn.Module):

    def __init__(self, isize=512, nz=100, nc=20, ndf=32):
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

        assert isize == 512

        model = OrderedDict()

        out_feat = ndf

        for i in range(1):
            model[f'adjacency_conv_{i}'] = AdjacencyConv(nc, out_feat)
            model[f'conv_{i}'] = nn.Conv1d(out_feat, out_feat, 4, 2, 1, bias=False)
            model[f'leaky_relu_{i}'] = nn.LeakyReLU(0.2)
            in_feat = out_feat

        for i in range(1, 4):
            out_feat = in_feat * 2
            model[f'adjacency_conv_{i}'] = AdjacencyConv(in_feat, in_feat)
            model[f'conv_{i}'] = nn.Conv1d(in_feat, out_feat, 4, 2, 1, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'leaky_relu_{i}'] = nn.LeakyReLU(0.2)
            in_feat = out_feat

        for i in range(4, 7):  # 7: int(np.log2(isize) - 2)
            out_feat = in_feat * 2
            model[f'conv_{i}'] = nn.Conv1d(in_feat, out_feat, 4, 2, 1, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'leaky_relu_{i}'] = nn.LeakyReLU(0.2)
            in_feat = out_feat

        assert in_feat == 2048, in_feat

        i += 1
        model[f'conv_{i}'] = nn.Conv1d(in_feat, 1, 4, 1, 0, bias=False)
        self.model = model
        for key in model:
            setattr(self, key, self.model[key])

    def forward(self, seq: Variable, adjs: List[Variable]):
        """
        Args:
            seq: A sequence of 512 amino acids (if your sequence is shorter, pad it with zeros).
                Shape: ``[?, 20, 512]``.
            adjs:
                Shape: ``[ [xxx, 512], [xxx, 256], [xxx, 128], [xxx, 64] ]``.
        """
        x = seq

        # 20 x 512
        for i in range(1):
            x = self.model[f'adjacency_conv_{i}'](x, adjs[i])
            x = self.model[f'conv_{i}'](x)
            x = self.model[f'leaky_relu_{i}'](x)

        # 32 x 256
        for i in range(1, 4):
            x = self.model[f'adjacency_conv_{i}'](x, adjs[i])
            x = self.model[f'conv_{i}'](x)
            x = self.model[f'instance_norm_{i}'](x)
            x = self.model[f'leaky_relu_{i}'](x)

        # 256 x 32
        for i in range(4, 7):
            x = self.model[f'conv_{i}'](x)
            x = self.model[f'instance_norm_{i}'](x)
            x = self.model[f'leaky_relu_{i}'](x)

        # 2048 x 4
        for i in range(7, 8):
            x = self.model[f'conv_{i}'](x)

        x = x.mean(0)
        return x.view(1)


class Generator(nn.Module):

    def __init__(self, isize=512, nz=100, nc=20, ngf=32):
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

        assert isize == 512

        # Dynamically calculate the input dimension (cngf)
        # cngf = 2048 for default inputs
        cngf = ngf // 2
        tisize = 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        model = OrderedDict()

        in_feat = nz
        out_feat = cngf

        # 100 x 1
        for i in range(1):
            model[f'convt_{i}'] = nn.ConvTranspose1d(in_feat, out_feat, 4, 1, 0, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'relu_{i}'] = nn.ReLU()
            in_feat = out_feat

        # 2048 x 4
        for i in range(1, 4):
            out_feat = in_feat // 2
            model[f'convt_{i}'] = nn.ConvTranspose1d(in_feat, out_feat, 4, 2, 1, bias=False)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'relu_{i}'] = nn.ReLU()
            in_feat = out_feat

        # 256 x 32
        for i in range(4, 6):
            out_feat = in_feat // 2
            model[f'convt_{i}'] = nn.ConvTranspose1d(in_feat, out_feat, 4, 2, 1, bias=False)
            model[f'adjacency_conv_{i}'] = AdjacencyConv(out_feat, out_feat)
            model[f'instance_norm_{i}'] = nn.InstanceNorm1d(out_feat, affine=False)
            model[f'relu_{i}'] = nn.ReLU()
            in_feat = out_feat

        # 64 x 128
        for i in range(6, 7):
            out_feat = in_feat // 2
            model[f'convt_{i}'] = nn.ConvTranspose1d(in_feat, out_feat, 4, 2, 1, bias=False)
            model[f'adjacency_conv_{i}'] = AdjacencyConv(out_feat, out_feat)
            model[f'relu_{i}'] = nn.Softmax(1)
            in_feat = out_feat

        assert in_feat == 32, in_feat

        # 32 x 256
        for i in range(7, 8):
            out_feat = in_feat // 2
            model[f'convt_{i}'] = nn.ConvTranspose1d(in_feat, nc, 4, 2, 1, bias=False)
            model[f'adjacency_conv_{i}'] = AdjacencyConv(nc, nc)
            model[f'softmax_{i}'] = nn.Softmax(1)

        self.model = model
        for key in model:
            setattr(self, key, self.model[key])

    def forward(self, z: Variable, adjs: List[Variable]):
        x = z
        adjs = list(reversed(adjs))

        # 100 x 1
        for i in range(1):
            x = self.model[f'convt_{i}'](x)
            x = self.model[f'instance_norm_{i}'](x)
            x = self.model[f'relu_{i}'](x)

        # 2048 x 4
        for i in range(1, 4):
            x = self.model[f'convt_{i}'](x)
            x = self.model[f'instance_norm_{i}'](x)
            x = self.model[f'relu_{i}'](x)

        # 256 x 32
        for i in range(4, 6):
            x = self.model[f'convt_{i}'](x)
            x = self.model[f'adjacency_conv_{i}'](x, adjs[i - 4])
            x = self.model[f'instance_norm_{i}'](x)
            x = self.model[f'relu_{i}'](x)

        # 64 x 128
        for i in range(6, 7):
            x = self.model[f'convt_{i}'](x)
            x = self.model[f'adjacency_conv_{i}'](x, adjs[i - 4])
            x = self.model[f'relu_{i}'](x)

        # 32 x 256
        for i in range(7, 8):
            x = self.model[f'convt_{i}'](x)
            x = self.model[f'adjacency_conv_{i}'](x, adjs[i - 4])
            x = self.model[f'softmax_{i}'](x)

        return x
