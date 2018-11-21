import numpy as np
import torch
from scipy import sparse

from pagnn import settings


def to_sparse_tensor(sparray: sparse.spmatrix) -> torch.sparse.FloatTensor:
    """Convert a scipy `spmatrix` into a torch sparse tensor (possibly on CUDA)."""
    indices = torch.LongTensor(np.vstack([sparray.row, sparray.col]))
    values = torch.FloatTensor(sparray.data)
    size = torch.Size(sparray.shape)
    tensor = torch.sparse_coo_tensor(indices, values, size)
    return tensor


def argmax_onehot(seq: torch.FloatTensor) -> torch.IntTensor:
    idx1 = torch.arange(0, seq.shape[0] * seq.shape[2], dtype=torch.long, device=settings.device)
    # Note: Takes the first value in case of duplicates.
    idx2 = seq.max(1)[1].view(-1)
    mat = torch.zeros(seq.shape[0] * seq.shape[2], seq.shape[1], device=settings.device)
    mat[idx1, idx2] = 1
    return mat.view(seq.shape[0], seq.shape[2], seq.shape[1]).transpose(1, 2).contiguous()


def remove_eye_sparse_tensor(
    x: torch.sparse.FloatTensor, bandwidth: int
) -> torch.sparse.FloatTensor:
    """Set diagonal (and offdiagonal) elements to zero.

    Args:
        x: Input array.
        bandwidth: Width of the diagonal 0 band.
    """
    if not bandwidth:
        return x
    indices = x._indices()
    values = x._values()
    keep_mask = (indices[0, :] - indices[1, :]).abs() > bandwidth
    out = torch.sparse_coo_tensor(indices[:, keep_mask], values[keep_mask])
    return out


def add_eye_sparse_tensor(x: torch.sparse.FloatTensor, bandwidth: int) -> torch.sparse.FloatTensor:
    """Not used!"""
    if not bandwidth:
        return x
    indices_list = [x._indices()]
    values_list = [x._values()]
    # Add diagonal
    indices_list.append(torch.arange(0, x.shape[1], dtype=torch.long).repeat(2, 1))
    values_list.append(torch.ones(x.shape[1], dtype=torch.float))
    # Add off-diagonals
    for k in range(1, bandwidth, 1):
        # Indices
        indices_upper = torch.stack(
            [
                torch.arange(x.shape[1] - k, dtype=torch.long),
                k + torch.arange(x.shape[1] - k, dtype=torch.long),
            ]
        )
        indices_lower = torch.stack(
            [
                k + torch.arange(x.shape[1] - k, dtype=torch.long),
                torch.arange(x.shape[1] - k, dtype=torch.long),
            ]
        )
        indices_list.extend([indices_upper, indices_lower])
        # Values
        values_upper = torch.ones(x.shape[1] - k, dtype=torch.float)
        values_lower = torch.ones(x.shape[1] - k, dtype=torch.float)
        values_list.extend([values_upper, values_lower])
    out = torch.sparse_coo_tensor(torch.cat(indices_list), torch.cat(values_list), size=x.size)
    return out


def expand_adjacency_tensor(adj: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
    row, col = adj._indices()
    num_interactions = len(row) * 2

    row_new = torch.cat(
        [
            torch.arange(0, num_interactions, 2, dtype=torch.long),
            torch.arange(1, max(1, num_interactions), 2, dtype=torch.long),
        ]
    )
    col_new = torch.cat([row, col])
    data_new = torch.ones(num_interactions, dtype=torch.float)
    size_new = (num_interactions, adj.shape[0])  # number of interactions x sequence length
    adj_new = torch.sparse_coo_tensor(torch.stack([row_new, col_new]), data_new, size=size_new)
    return adj_new
