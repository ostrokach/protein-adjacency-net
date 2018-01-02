"""Basic single hidden layer neural networks."""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pagnn import GAP_LENGTH


class SingleDomainNet(nn.Module):
    """A neural network that takes a single domain and adjacency matrix as input."""

    def __init__(self):
        super().__init__()
        n_filters = 12
        self.spatial_conv = nn.Conv1d(20, n_filters, 2, stride=2, bias=False)
        self.combine_convs = nn.Linear(n_filters, 1, bias=False)

    def forward(self, aa, adjacency):
        x = aa @ adjacency.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @ adjacency[::2, :]
        # import pdb; pdb.set_trace()
        # x = x.sum(dim=0) / adjacency.sum(dim=0)
        x, idxs = (x / adjacency.sum(dim=0)).max(dim=2)
        x = self.combine_convs(x)
        x = x.squeeze()
        return F.sigmoid(x)


class MultiDomainNet(nn.Module):
    """A neural network that takes *multiple* domains and an adjacency matrix as input."""

    def __init__(self):
        super().__init__()
        n_aa = 20
        n_filters = 12
        self.spatial_conv = nn.Conv1d(n_aa, n_filters, 2, stride=2, bias=False)
        self.combine_convs = nn.Linear(n_filters, 1, bias=False)

    def forward(self, aa: Variable, adjs: Variable):
        """Forward pass through the network.

        Args:
            aa: PyTorch `Variable` containing the input sequence.
                Size: [batch size (2), number of amino acids (20), sequence length].
            adjs: PyTorch `Variable` containing the adjacency matrix sequence.
                Size: [number of contacts * 2, sequence length].

        Returns:
        """
        # Apply convolutions in interaction space
        aai = self._expand(aa, adjs)
        aai = self.spatial_conv(aai)
        aa = self._contract(aai, adjs)
        # Aggregate by domain
        domain_scores = self._agg_by_domain(aa, adjs)
        return F.sigmoid(domain_scores)

    def _expand(self, aa: Variable, adjs: List[Variable]) -> Variable:
        aai_list = []
        start = 0
        for adj in adjs:
            end = start + adj.size()[1]
            aai = aa[:, :, start:end] @ adj.transpose(0, 1)
            aai_list.append(aai)
            start = end + GAP_LENGTH
        assert aa.size()[2] == end, (aa.size(), end)
        return torch.cat(aai_list, dim=2)

    def _contract(self, aai: Variable, adjs: List[Variable]) -> Variable:
        aa_list = []
        start = 0
        for adj in adjs:
            end = start + adj.size()[0]
            aa = aai[:, :, start:end] @ adj[::2, :]
            aa_list.append(aa)
            start = end + GAP_LENGTH
        assert aai.size()[2] == end, (aai.size(), end)
        return torch.cat(aa_list, dim=2)

    def _agg_by_domain(self, aa: Variable, adjs: List[Variable]) -> Variable:
        domain_scores = []
        start = 0
        for adj in adjs:
            end = start + adj.size()[1]
            aa_domain = aa[:, :, start:end]
            aa_domain_max, idxs = (aa_domain / adj.sum(dim=0)).max(dim=2)
            aa_domain_final = self.combine_convs(aa_domain_max).squeeze()
            domain_scores.append(aa_domain_final)
            start = end + GAP_LENGTH
        assert aa.size()[2] == end, (aa.size(), end)
        return torch.cat(domain_scores)
