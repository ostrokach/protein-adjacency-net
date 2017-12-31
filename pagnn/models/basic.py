"""Basic single hidden layer neural networks."""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
        n_filters = 12
        self.spatial_conv = nn.Conv1d(20, n_filters, 2, stride=2, bias=False)
        self.combine_convs = nn.Linear(n_filters, 1, bias=False)

    def forward(self, aa: Variable, adjacency: Variable, domain_lengths: List[int]):
        """Forward pass through the network.

        Args:
            aa: PyTorch `Variable` containing the input sequence.
                Size: [batch size (2), number of amino acids (20), sequence length].
            adjacency: PyTorch `Variable` containing the adjacency matrix sequence.
                Size: [number of contacts * 2, sequence length].
            domain_ranges: List of ranges for the different domains in `aa` and `adjacency`.

        Returns:
        """
        x = aa @ adjacency.transpose(0, 1)
        x = self.spatial_conv(x)
        x = x @ adjacency[::2, :]
        domains = []
        cur_idx = 0
        for domain_len in domain_lengths:
            domain = x[:, :, cur_idx:cur_idx + domain_len]
            # import pdb; pdb.set_trace()
            domain, idx = (
                domain / adjacency[:, cur_idx:cur_idx + domain_len].sum(dim=0)).max(dim=2)
            domain = self.combine_convs(domain)
            domain = domain.squeeze()
            domains.append(domain)
            cur_idx += domain_len
        return F.sigmoid(torch.cat(domains))
