from typing import List

import numpy as np
import logging

logger = logging.getLogger(__name__)

_AMINO_ACIDS: List[str] = [
    'G', 'V', 'A', 'L', 'I', 'C', 'M', 'F', 'W', 'P', 'D', 'E', 'S', 'T', 'Y', 'Q', 'N', 'K', 'R',
    'H'
]

_AMINO_ACIDS_BYTEARRAY = bytearray(''.join(_AMINO_ACIDS).encode())


def get_seq_array(seq: bytes) -> np.ndarray:
    """Convert amino acid sequence into a one-hot encoded array.

    Args:
        seq: Amino acid sequence.

    Returns:
        Numpy array containing the one-hot encoding of the amino acid sequence.
    """
    amino_acids = _AMINO_ACIDS_BYTEARRAY
    seq_array = np.zeros((20, len(seq)))
    for i, aa in enumerate(seq):
        try:
            seq_array[amino_acids.index(aa), i] = 1
        except ValueError as e:
            seq_array[:, i] = 1 / 20
            if aa not in bytearray(b'X'):
                logger.debug("Could not convert the following residue to one-hot encoding: %s",
                             chr(aa))
    return seq_array


def get_adjacency(seq_len: int, adjacency_idx_1: np.array, adjacency_idx_2: np.array) -> np.array:
    """Construct an adjacency matrix from the data available for each row in the DataFrame.

    Args:
        seq_len:
        adjacency_idx_1: Indexes of the residues that are involved in intrachain interactions.
        adjacency_idx_2: Indexes of the residues that are involved in intrachain interactions.

    Returns:
        An adjacency matrix for the given sequence `qseq`.
    """
    na_mask = (np.isnan(adjacency_idx_1) | np.isnan(adjacency_idx_2))
    if na_mask.any():
        logger.debug("Removing %s null indices.", na_mask.sum())
        adjacency_idx_1 = np.array(adjacency_idx_1[~na_mask], dtype=np.int_)
        adjacency_idx_2 = np.array(adjacency_idx_2[~na_mask], dtype=np.int_)

    # There are no nulls, so should be safe to convert to integers now
    adjacency_idx_1 = adjacency_idx_1.astype(np.int_)
    adjacency_idx_2 = adjacency_idx_2.astype(np.int_)

    too_close_mask = np.abs(adjacency_idx_1 - adjacency_idx_2) <= 2
    if too_close_mask.any():
        logger.debug("Removing %s too close indices.", na_mask.sum())
        adjacency_idx_1 = np.array(adjacency_idx_1[~too_close_mask])
        adjacency_idx_2 = np.array(adjacency_idx_2[~too_close_mask])

    adj = np.eye(seq_len, dtype=np.bool_)
    adj[adjacency_idx_1, adjacency_idx_2] = 1
    assert (adj == adj.T).all().all(), adj
    return adj


def expand_adjacency(adj: np.array) -> np.array:
    """Convert adjacency matrix into a strided mask.

    Args:
        adj: Adjacency matrix.

    Returns:
        Adjacency matrix converted to *two-rows-per-interaction* format.
    """
    new_adj = np.zeros((int(adj.sum() * 2), adj.shape[1]), dtype=adj.dtype)
    a, b = adj.nonzero()
    a_idx = np.arange(0, len(a) * 2, 2)
    b_idx = np.arange(1, len(b) * 2, 2)
    new_adj[a_idx, a] = 1
    new_adj[b_idx, b] = 1
    assert (new_adj.sum(axis=1) == 1).all(), new_adj
    return new_adj
    # idx = 0
    # for x, y in zip(*adj.nonzero()):
    #     new_adj[idx, x] = 1
    #     new_adj[idx + 1, y] = 1
    #     idx += 2
    # return new_adj


def get_seq_identity(seq: bytes, other_seq: bytes) -> float:
    """Return the fraction of amino acids that are the same in `seq` and `other_seq`.

    Examples:
        >>> get_seq_identity(b'AAGC', b'AACC')
        0.75
    """
    assert len(seq) == len(other_seq)
    return sum(a == b for a, b in zip(seq, other_seq)) / len(seq)


def get_adj_identity(adj, other_adj, min_distance=3) -> float:
    """Return the fraction of (distant) contacts that are the same in the two adjacency matrices.

    Examples:
        >>> adj = np.eye(5) + np.eye(5, k=2) + np.eye(5, k=-2) + np.eye(5, k=4) + np.eye(5, k=-4)
        >>> other_adj =  np.eye(5) + np.eye(5, k=2) + np.eye(5, k=-2)
        >>> get_adj_identity(adj, other_adj, 2)
        0.75
    """
    ones = (adj == 1) | (other_adj == 1)
    for k in range(min_distance):
        ones &= (~np.eye(adj.shape[0], k=k, dtype=bool) & ~np.eye(adj.shape[0], k=-k, dtype=bool))

    if ones.sum() == 0:
        return 0

    same = {tuple(x) for x in zip(*np.where(ones & (adj == other_adj)))}
    different = {tuple(x) for x in zip(*np.where(ones & (adj != other_adj)))}
    assert not same & different
    assert len(same) + len(different) == ones.sum()

    return len(same) / ones.sum()
