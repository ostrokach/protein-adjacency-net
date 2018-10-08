import logging
from typing import List

import numpy as np
import torch
from numba import jit
from scipy import sparse

logger = logging.getLogger(__name__)

# Warning: Do not change the order of amino acids without chaning the order of
# `AMINO_ACIDS_BYTES` in `_seq_to_array`!
AMINO_ACIDS: List[str] = [
    "G",
    "V",
    "A",
    "L",
    "I",
    "C",
    "M",
    "F",
    "W",
    "P",
    "D",
    "E",
    "S",
    "T",
    "Y",
    "Q",
    "N",
    "K",
    "R",
    "H",
]


def seq_to_array(seq: bytes) -> torch.sparse.FloatTensor:
    """Convert amino acid sequence into a one-hot encoded array.

    Args:
        seq: Amino acid sequence.

    Returns:
        Numpy array containing the one-hot encoding of the amino acid sequence.
    """
    x_idxs, y_idxs, data = _seq_to_array(seq)
    indices = torch.tensor([x_idxs, y_idxs])
    values = torch.tensor(data, dtype=torch.float32)
    seq_tensor = torch.sparse_coo_tensor(indices, values, size=(20, len(seq)))
    # import pdb

    # pdb.set_trace()
    assert seq_tensor.shape[1] == seq_tensor._indices().shape[1] == seq_tensor._values().shape[0]
    return seq_tensor


def array_to_seq(array: np.ndarray) -> str:
    max_idxs = np.argmax(array, 0)
    seq = "".join(AMINO_ACIDS[i] for i in max_idxs)
    return seq


@jit(nopython=True)
def _seq_to_array(seq: bytes) -> sparse.spmatrix:
    amino_acids = [71, 86, 65, 76, 73, 67, 77, 70, 87, 80, 68, 69, 83, 84, 89, 81, 78, 75, 82, 72]

    data = []
    x_idxs = []
    y_idxs = []
    skip_char = 46  # ord('.')
    for y, aa in enumerate(seq):
        if aa == skip_char:
            # We use '.' for padding sequences with 0s
            # Putting this into `except` makes it much slower
            continue
        match = False
        for x, aa_ref in enumerate(amino_acids):
            if aa == aa_ref:
                # x = amino_acids.index(aa)
                x_idxs.append(x)
                y_idxs.append(y)
                data.append(1)
                match = True
                break
        if not match:
            # Add an empty value to keep _indexes() and _values() the right length
            x_idxs.append(0)
            y_idxs.append(y)
            data.append(0)
    return x_idxs, y_idxs, data


def get_adjacency(
    seq_len: int, adjacency_idx_1: np.array, adjacency_idx_2: np.array
) -> sparse.spmatrix:
    """Construct an adjacency matrix from the data available for each row in the DataFrame.

    Args:
        seq_len:
        adjacency_idx_1: Indexes of the residues that are involved in intrachain interactions.
        adjacency_idx_2: Indexes of the residues that are involved in intrachain interactions.

    Returns:
        An adjacency matrix for the given sequence `qseq`.
    """
    na_mask = np.isnan(adjacency_idx_1) | np.isnan(adjacency_idx_2)
    if na_mask.any():
        logger.debug("Removing %s null indices.", na_mask.sum())
        adjacency_idx_1 = np.array(adjacency_idx_1[~na_mask], dtype=np.int_)
        adjacency_idx_2 = np.array(adjacency_idx_2[~na_mask], dtype=np.int_)

    # There are no nulls, so should be safe to convert to integers now
    adjacency_idx_1 = adjacency_idx_1.astype(np.int_)
    adjacency_idx_2 = adjacency_idx_2.astype(np.int_)

    # TODO(AS): Make sure that this is still what we want!
    # too_close_mask = (np.abs(adjacency_idx_1 - adjacency_idx_2) <= 1)
    # if too_close_mask.any():
    #     logger.debug("Removing %s too close indices.", too_close_mask.sum())
    #     adjacency_idx_1 = adjacency_idx_1[~too_close_mask]
    #     adjacency_idx_2 = adjacency_idx_2[~too_close_mask]

    # # Add eye (we remove it later if neccessary)
    # adjacency_idx_1 = np.hstack([np.arange(seq_len), adjacency_idx_1])
    # adjacency_idx_2 = np.hstack([np.arange(seq_len), adjacency_idx_2])

    assert adjacency_idx_1.shape == adjacency_idx_2.shape

    adj = sparse.coo_matrix(
        (np.ones(len(adjacency_idx_1)), (adjacency_idx_1, adjacency_idx_2)),
        shape=(seq_len, seq_len),
        dtype=np.int16,
    )

    # Make sure that the matrix is symetrical
    idx1 = {(r, c) for r, c in zip(adj.row, adj.col)}
    idx2 = {(c, r) for r, c in zip(adj.row, adj.col)}
    assert not idx1 ^ idx2
    return adj


def expand_adjacency(adj: sparse.spmatrix) -> torch.sparse.FloatTensor:
    """Convert adjacency matrix into a strided mask.

    Args:
        adj: Adjacency matrix.

    Returns:
        Adjacency matrix converted to *two-rows-per-interaction* format.

    Examples:
        >>> from scipy import sparse
        >>> adj = sparse.coo_matrix((np.ones(3), (np.arange(3), np.arange(3)), ))
        >>> expanded_adj = expand_adjacency(adj)
        >>> expanded_adj.row
        array([0, 2, 4, 1, 3, 5], dtype=int32)
        >>> expanded_adj.col
        array([0, 1, 2, 0, 1, 2], dtype=int32)
    """
    # # Return scipy sparse array
    # row_idx = np.arange(0, len(adj.row) * 2, 2)
    # col_idx = np.arange(1, len(adj.col) * 2, 2)
    # new_adj = sparse.coo_matrix(
    #     (np.ones(len(adj.row) + len(adj.col)),
    #     (np.r_[row_idx, col_idx], np.r_[adj.row, adj.col])),
    #     dtype=np.float32,
    #     shape=(len(adj.data) * 2, adj.shape[0]),
    # )
    # assert (new_adj.sum(axis=1) == 1).all(), new_adj

    # (np.r_[0 : len(adj.row) * 2 : 2, 1 : len(adj.col) * 2 : 2], np.r_[adj.row, adj.col]),

    indices_row = torch.cat(
        [
            torch.arange(0, len(adj.row) * 2, 2, dtype=torch.long),
            torch.arange(1, max(1, len(adj.col) * 2), 2, dtype=torch.long),
        ]
    )
    # indices_col = adj._indices()
    indices_col = torch.cat(
        [torch.as_tensor(adj.row, dtype=torch.long), torch.as_tensor(adj.col, dtype=torch.long)]
    )
    new_adj = torch.sparse_coo_tensor(
        torch.stack([indices_row, indices_col]),
        torch.ones(len(adj.row) + len(adj.col), dtype=torch.float),
        size=(len(adj.data) * 2, adj.shape[0]),
    )

    # TODO: Remove computationally-intensive assert
    # assert (new_adj.to_dense().sum(1) == 1).all()
    return new_adj
    # idx = 0
    # for x, y in zip(*adj.nonzero()):
    #     new_adj[idx, x] = 1
    #     new_adj[idx + 1, y] = 1
    #     idx += 2
    # return new_adj


def get_seq_identity_bytes(seq: bytes, other_seq: bytes) -> float:
    """Return the fraction of amino acids that are the same in `seq` and `other_seq`.

    Examples:
        >>> get_seq_identity(b'AAGC', b'AACC')
        0.75
    """
    assert len(seq) == len(other_seq)
    return sum(a == b for a, b in zip(seq, other_seq)) / len(seq)


def get_seq_identity(seq: torch.Tensor, other_seq: torch.Tensor) -> torch.Tensor:
    """Return the fraction of amino acids that are the same in `seq` and `other_seq`.

    Examples:
        >>> get_seq_identity(b'AAGC', b'AACC')
        0.75
    """
    num_equal = (seq._indices()[0, :] == other_seq._indices()[0, :]).sum().to(torch.float)
    num_total = seq.size()[1]
    return (num_equal / num_total).item()


def get_adj_identity(adj: sparse.spmatrix, other_adj: sparse.spmatrix, min_distance=3) -> float:
    """Return the fraction of (distant) contacts that are the same in the two adjacency matrices.

    Examples:
        >>> from scipy import sparse
        >>> adj = sparse.coo_matrix((
        ...     np.ones(9),
        ...     (np.r_[range(7), [6, 0]],
        ...      np.r_[range(7), [0, 6]]), ))
        >>> other_adj = sparse.coo_matrix((
        ...     np.ones(11),
        ...     (np.r_[range(7), [5, 6, 0, 0]],
        ...      np.r_[range(7), [0, 0, 5, 6]]), ))
        >>> get_adj_identity(adj, other_adj, 2)
        0.5
    """
    adj_contacts = {tuple(x) for x in zip(adj.row, adj.col) if abs(x[0] - x[1]) >= min_distance}
    other_adj_contacts = {
        tuple(x) for x in zip(other_adj.row, other_adj.col) if abs(x[0] - x[1]) >= min_distance
    }
    if len(adj_contacts) == 0 or len(other_adj_contacts) == 0:
        return 0

    return len(adj_contacts & other_adj_contacts) / len(adj_contacts | other_adj_contacts)
