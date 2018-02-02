from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ModernNet(nn.Module):
    """A neural network that takes multiple ``(sequence, adjacency_matrix)`` tuples."""

    def __init__(self, n_filters: int = 64) -> None:
        super().__init__()
        n_aa = 20
        self.spatial_conv = nn.Conv1d(n_aa, n_filters, 2, stride=2, bias=False)
        self.combine_weights = nn.Linear(n_filters, 1, bias=False)

    def forward(self, pos: List[Tuple[Variable, Variable]],
                neg: List[Tuple[Variable, Variable]]) -> List[Variable]:
        assert len(neg) == 0 or len(pos) == 1
        # pos_seq = pos[0][0]
        pos_adj = pos[0][1]
        # inputs = pos + [(seq, pos[1]) for seq, _ in neg] + [(pos[0], adj) for _, adj in neg]
        inputs = pos + [(seq, pos_adj) for seq, _ in neg]
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
            seq = seq.unsqueeze(0)
            x = seq @ adj.transpose(0, 1)
            x = self.spatial_conv(x)
            x = x @ adj[::2, :]
            x = x / adj.sum(dim=0)
            x, idxs = x.max(dim=2)
            x = self.combine_weights(x)
            x = x.squeeze()
            scores.append(x)
        scores = torch.cat(scores)
        scores = F.sigmoid(scores)
        return scores
