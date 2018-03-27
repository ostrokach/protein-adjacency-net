"""Basic single hidden layer neural networks."""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pagnn import settings


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
        # x = x.sum(dim=0) / adjacency.sum(dim=0)
        x, idxs = (x / adjacency.sum(dim=0)).max(dim=2)
        x = self.combine_convs(x)
        x = x.squeeze()
        return F.sigmoid(x)


class MultiDomainNet(nn.Module):
    """A neural network that takes *multiple* domains and an adjacency matrix as input."""

    def __init__(self, n_filters=12):
        super().__init__()
        n_aa = 20
        self.spatial_conv = nn.Conv1d(n_aa, n_filters, 2, stride=2, bias=False)
        self.combine_convs = nn.Linear(n_filters, 1, bias=False)

    def forward(self, aa: Variable, adjs: Variable):
        """Forward pass through the network.

        Args:
            aa: PyTorch `Variable` containing the input sequence.
                Size: [batch size (2), number of amino acids (20), sequence length].
            adjs: PyTorch `Variable` containing the adjacency matrix sequence.
                Has to be created from `np.int16` or higher!!
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
            start = end + settings.GAP_LENGTH
        assert aa.size()[2] == end, (aa.size(), end)
        return torch.cat(aai_list, dim=2)

    def _contract(self, aai: Variable, adjs: List[Variable]) -> Variable:
        aa_list = []
        start = 0
        for adj in adjs:
            end = start + adj.size()[0] // 2
            aa = aai[:, :, start:end] @ adj[::2, :]
            aa_list.append(aa)
            start = end + settings.GAP_LENGTH
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
            start = end + settings.GAP_LENGTH
        assert aa.size()[2] == end, (aa.size(), end)
        return torch.cat(domain_scores)


class MultiDomainNetNew(nn.Module):
    """A neural network that takes *multiple* domains and an adjacency matrix as input."""

    def __init__(self, n_filters=12):
        super().__init__()
        n_aa = 20
        self.spatial_conv = nn.Conv1d(n_aa, n_filters, 2, stride=2, bias=False)
        self.combine_convs = nn.Linear(n_aa + n_filters, 1, bias=False)

    def forward(self, aa: Variable, adjs: Variable):
        """Forward pass through the network.

        Args:
            aa: PyTorch `Variable` containing the input sequence.
                Size: [batch size (2), number of amino acids (20), sequence length].
            adjs: PyTorch `Variable` containing the adjacency matrix sequence.
                Size: [number of contacts * 2, sequence length].

        Returns:
        """
        aa_in = aa
        print(f"aa_in: {count_nans(aa_in)}")
        # Apply convolutions in interaction space
        aai = self._expand(aa, adjs)
        print(f"aai1: {count_nans(aai)}")
        aai = self.spatial_conv(aai)
        print(f"aai2: {count_nans(aai)}")
        aa = self._contract(aai, adjs)
        print(f"aa: {count_nans(aa)}")

        # Aggregate by domain
        domain_scores = self._agg_by_domain(aa, aa_in, adjs)
        # domain_scores = F.sigmoid(domain_scores)
        return domain_scores

    def _expand(self, aa: Variable, adjs: List[Variable]) -> Variable:
        aai_list = []
        start = 0
        for i, adj in enumerate(adjs):
            end = start + adj.size()[1]
            aai = aa[:, :, start:end] @ adj.transpose(0, 1)
            if count_nans(adj) or count_nans(aai):
                print(f"adj ({i}): {count_nans(adj)}")
                print(f"aai ({i}): {count_nans(aai)}")
                print(adj)
                print(aai)
            aai_list.append(aai)
            start = end + settings.GAP_LENGTH
        assert aa.size()[2] == end, (aa.size(), end)
        return torch.cat(aai_list, dim=2)

    def _contract(self, aai: Variable, adjs: List[Variable]) -> Variable:
        aa_list = []
        start = 0
        for i, adj in enumerate(adjs):
            end = start + adj.size()[0] // 2
            aa = aai[:, :, start:end] @ adj[::2, :]
            if count_nans(adj) or count_nans(aa):
                print(f"adj ({i}): {count_nans(adj)}")
                print(f"aa ({i}): {count_nans(aa)}")
                print(adj)
                print(aa)
            aa_list.append(aa)
            start = end + settings.GAP_LENGTH
        assert aai.size()[2] == end, (aai.size(), end)
        return torch.cat(aa_list, dim=2)

    def _agg_by_domain(self, aa: Variable, aa_in: Variable, adjs: List[Variable]) -> Variable:
        domain_scores = []
        start = 0
        for adj in adjs:
            end = start + adj.size()[1]
            aa_domain = aa[:, :, start:end]
            aa_domain = (aa_domain / adj.sum(dim=0))

            print(f"aa_domain: {count_nans(aa_domain)}")
            aa_in_domain = aa_in[:, :, start:end]
            aa_combined = torch.cat([aa_in_domain, aa_domain], dim=1)
            aa_combined_max, idx = aa_combined.max(dim=2)
            print(f"aa_combined_max: {count_nans(aa_combined_max)}")

            aa_domain_final = self.combine_convs(aa_combined_max).squeeze()
            print(f"aa_domain_final: {count_nans(aa_domain_final)}")

            domain_scores.append(aa_domain_final)
            start = end + settings.GAP_LENGTH
        assert aa.size()[2] == end, (aa.size(), end)
        return torch.cat(domain_scores)


def count_nans(x):
    import numpy as np
    return np.isnan(x.data.numpy()).sum()
