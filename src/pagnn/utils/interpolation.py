from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse


def interpolate_sequences(
        positive_seq: bytes,
        negative_seq: bytes,
        interpolate: int = 0,
        random_state: Optional[np.random.RandomState] = None) -> Tuple[List[bytes], List[float]]:
    """
    Examples:
        >>> interpolated_seqs, interpolated_targets = interpolate_sequences(
        ...     b'AAAAA', b'BBBBB', 4, np.random.RandomState(42))
        >>> interpolated_seqs
        [b'AAABA', b'BAAAB', b'AABBB', b'ABBBB']
        >>> [round(f, 3) for f in interpolated_targets]
        [0.8, 0.6, 0.4, 0.2]
    """
    if random_state is None:
        random_state = np.random.RandomState()
    interpolated_seqs = []
    fraction_positive = np.linspace(1, 0, interpolate + 2)[1:-1]
    if interpolate:
        for i in range(interpolate):
            fp = fraction_positive[i]
            idx = random_state.choice(
                np.arange(len(positive_seq)), int(round(fp * len(positive_seq))), replace=False)
            seq = bytearray(negative_seq)
            for i in idx:
                seq[i] = positive_seq[i]
            interpolated_seqs.append(bytes(seq))
    return interpolated_seqs, fraction_positive.tolist()


def interpolate_adjacencies(positive_adj: sparse.spmatrix,
                            negative_adj: sparse.spmatrix,
                            interpolate: int = 0,
                            random_state: Optional[np.random.RandomState] = None
                           ) -> Tuple[List[sparse.spmatrix], List[float]]:
    """

    Examples:
        >>> from scipy import sparse
        >>> positive_adj = sparse.coo_matrix((
        ...     np.ones(11), (
        ...         np.r_[np.arange(7), [0, 0, 5, 6]],
        ...         np.r_[np.arange(7), [5, 6, 0, 0]]), ))
        >>> negative_adj = sparse.coo_matrix((np.ones(7), (np.arange(7), np.arange(7)), ))
        >>> interpolated_adjs, interpolated_targets = interpolate_adjacencies(
        ...     positive_adj, negative_adj, 1, np.random.RandomState(42))
        >>> interpolated_targets
        [0.5]
        >>> interpolated_adjs[0].row
        array([0, 0, 1, 2, 3, 4, 5, 6, 6], dtype=int32)
        >>> interpolated_adjs[0].col
        array([0, 6, 1, 2, 3, 4, 5, 0, 6], dtype=int32)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    interpolated_adjs = []
    fraction_positive = np.linspace(1, 0, interpolate + 2)[1:-1]
    if interpolate:
        # TODO: This can be sped up significantly if we actually use it.
        positive_adj = positive_adj.tocsr()
        negative_adj = negative_adj.tocsr()
        mismatches = np.array(np.where((positive_adj != negative_adj).todense()))
        mismatches = mismatches[:, mismatches[0, :] <= mismatches[1, :]]
        for i in range(interpolate):
            fp = fraction_positive[i]
            idxs = random_state.choice(
                np.arange(mismatches.shape[0]), int(round(mismatches.shape[0] * fp)), replace=False)
            adj = negative_adj.tocsr()
            idx_1, idx_2 = mismatches[:, idxs]
            adj[idx_1, idx_2] = positive_adj[idx_1, idx_2]
            adj[idx_2, idx_1] = positive_adj[idx_2, idx_1]
            interpolated_adjs.append(adj.tocoo())
    return interpolated_adjs, fraction_positive.tolist()
