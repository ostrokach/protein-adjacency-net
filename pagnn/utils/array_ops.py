import math

import numpy as np
import torch
from numba import jit
from scipy import sparse

from pagnn import settings


def to_sparse_tensor(sparray: sparse.spmatrix) -> torch.sparse.FloatTensor:
    """Convert a scipy `spmatrix` into a torch sparse tensor (possibly on CUDA)."""
    if sparray.nnz == 0:
        i = torch.LongTensor()
        v = torch.FloatTensor()
    else:
        i = torch.LongTensor(np.vstack([sparray.row, sparray.col]))
        v = torch.FloatTensor(sparray.data)
    s = torch.Size(sparray.shape)
    tensor = torch.sparse.FloatTensor(i, v, s)
    return tensor


def argmax_onehot(seq: torch.FloatTensor) -> torch.IntTensor:
    idx1 = torch.arange(0, seq.shape[0] * seq.shape[2], dtype=torch.long, device=settings.device)
    # Note: Takes the first value in case of duplicates.
    idx2 = seq.max(1)[1].view(-1)
    mat = torch.zeros(seq.shape[0] * seq.shape[2], seq.shape[1], device=settings.device)
    mat[idx1, idx2] = 1
    return mat.view(seq.shape[0], seq.shape[2], seq.shape[1]).transpose(1, 2).contiguous()


@jit(nopython=True)
def conv1d_shape(in_channels, kernel_size, stride=1, padding=0, dilation=1):
    return math.floor((in_channels + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


@jit(nopython=True)
def conv1d_shape_ceil(in_channels, kernel_size, stride=1, padding=0, dilation=1):
    return math.ceil((in_channels + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


@jit(nopython=True)
def conv2d_shape(shape, kernel_size, stride=1, padding=0, dilation=1):
    """
    Note:
        Actual convolutions in PyTorch (e.g. `nn.Conv1d`), round down, not up.
    """
    out_shape = (
        conv1d_shape(
            shape[0], kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        ),
        conv1d_shape(
            shape[1], kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        ),
    )
    return out_shape


def remove_eye(x: np.ndarray, bandwidth: int, copy: bool = True) -> np.ndarray:
    """Set diagonal (and offdiagonal) elements to zero.

    Args:
        x: Input array.
        bandwidth: Width of the diagonal 0 band.
        copy: Whether the returned array should be a copy of the original
            (ignored if `bandwidth` is falsy).
    """
    if not bandwidth:
        return x
    if copy:
        x = x.copy()
    for k in range(-bandwidth + 1, bandwidth, 1):
        x[np.eye(x.shape[0], k=k, dtype=bool)] = 0
    return x


def remove_eye_sparse(x: sparse.spmatrix, bandwidth: int, copy: bool = True) -> sparse.spmatrix:
    """Set diagonal (and offdiagonal) elements to zero.

    Args:
        x: Input array.
        bandwidth: Width of the diagonal 0 band.
        copy: Whether the returned array should be a copy of the original
            (ignored if `bandwidth` is falsy).
    """
    if not bandwidth:
        return x
    if copy:
        x = x.copy()
    keep_mask = np.ones(len(x.data), dtype=bool)
    for i, (r, c) in enumerate(zip(x.row, x.col)):
        if abs(r - c) <= (bandwidth - 1):
            keep_mask[i] = 0
    x.data = x.data[keep_mask]
    x.row = x.row[keep_mask]
    x.col = x.col[keep_mask]
    return x


def add_eye_sparse(x: sparse.spmatrix, bandwidth: int, copy: bool = True) -> sparse.spmatrix:
    if not bandwidth:
        return x
    if copy:
        x = x.copy()
    # Add diagonal
    x.data = np.hstack([x.data, np.ones(x.shape[1])])
    x.row = np.hstack([x.row, np.arange(x.shape[1])])
    x.col = np.hstack([x.col, np.arange(x.shape[1])])
    # Add off-diagonals
    for k in range(1, bandwidth, 1):
        x.data = np.hstack([x.data, np.ones(x.shape[1] - k), np.ones(x.shape[1] - k)])
        x.row = np.hstack([x.row, np.arange(x.shape[1] - k), k + np.arange(x.shape[1] - k)])
        x.col = np.hstack([x.col, k + np.arange(x.shape[1] - k), np.arange(x.shape[1] - k)])
    return x


def reshape_internal_dim(x, dim, size):
    # Easy to implement for more dimensions at the cost of readability
    if x.shape[dim] == size:
        return x
    assert len(x.shape) == 3
    out = x.transpose(dim, -1).contiguous().reshape(x.shape[0], -1, size)
    back = out.transpose(-1, dim).contiguous()
    return back


def unfold_to(x, length):
    return x.transpose(1, 2).contiguous().reshape(x.shape[0], -1, length)


def unfold_from(x, length):
    return x.view(x.shape[0], -1, length).transpose(2, 1).contiguous()


def padding_amount(x, length):
    return int((length - (np.prod(x.shape[1:]) % length)) // x.shape[1])
