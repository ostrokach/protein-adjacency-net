from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pagnn import DataVarCollection


class ModernNet(nn.Module):
    """A neural network that takes multiple ``(sequence, adjacency_matrix)`` tuples."""

    def __init__(self, n_filters: int = 64) -> None:
        super().__init__()
        n_aa = 20
        self.spatial_conv = nn.Conv1d(n_aa, n_filters, 2, stride=2, bias=False)
        self.combine_weights = nn.Linear(n_filters, 1, bias=False)

    def forward(self, dvc: DataVarCollection) -> List[Variable]:
        pos, neg = dvc
        assert len(neg) == 0 or len(pos) == 1
        keep_pos_adj = [(seq, pos[0][1]) for seq, _ in neg if seq is not None]
        keep_pos_seq = [(pos[0][0], adj) for _, adj in neg if adj is not None]
        inputs = pos + keep_pos_adj + keep_pos_seq
        return self._forward(inputs)

    def _forward(self, inputs: List[Tuple[Variable, Variable]]) -> List[Variable]:
        """Forward pass through the network.

        Args:
            inputs: List of ``(sequence, adjacency_matrix)`` tuples.

                - `sequence` has size: ``[number of amino acids (20), sequence length]``.
                - `adjacency_matrix` has size ``[number of contacts * 2, sequence length]``.

        Returns:
            List of scores in range ``[0, 1)``.
        """
        scores = []
        for seq, adj in inputs:
            x = seq @ adj.transpose(0, 1)
            x = self.spatial_conv(x)
            x = x @ adj[::2, :]
            x = x / adj.sum(dim=0)
            x, idxs = x.max(dim=2)
            x = self.combine_weights(x)
            # x = x.squeeze()
            scores.append(x)
        scores = torch.cat(scores)
        scores = F.sigmoid(scores)
        return scores
