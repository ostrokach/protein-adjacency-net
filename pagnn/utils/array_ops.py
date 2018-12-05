import math

import numpy as np
import torch
from numba import jit
from scipy import sparse

from pagnn.types import SparseMat


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


def remove_eye_sparse(sp: SparseMat, bandwidth: int) -> SparseMat:
    """Set diagonal (and offdiagonal) elements to zero.

    Args:
        x: Input array.
        bandwidth: Width of the diagonal 0 band.
        copy: Whether the returned array should be a copy of the original
            (ignored if `bandwidth` is falsy).
    """
    if not bandwidth:
        return sp
    keep_mask = abs(sp.indices[0, :] - sp.indices[1, :]) >= bandwidth
    indices = sp.indices[:, keep_mask]
    values = sp.values[keep_mask]
    return sp._replace(indices=indices, values=values)


def add_eye_sparse(sp: sparse.spmatrix, bandwidth: int) -> sparse.spmatrix:
    if not bandwidth:
        return sp
    row_list = [sp.indices[0, :]]
    col_list = [sp.indices[1, :]]
    data_list = [sp.values]
    n = sp.n
    # Add diagonal
    row_list.append(torch.arange(n, dtype=torch.long))
    col_list.append(torch.arange(n, dtype=torch.long))
    data_list.append(torch.ones(n, dtype=torch.float))
    # Add off-diagonals
    for k in range(1, bandwidth, 1):
        data_list.extend([np.ones(n - k), np.ones(n - k)])
        row_list.extend([np.arange(n - k), k + np.arange(n - k)])
        col_list.extend([k + np.arange(n - k), np.arange(n - k)])
    row = np.hstack(row_list)
    col = np.hstack(col_list)
    values = np.hstack(data_list)
    return sp._replace(indices=torch.stack([row, col]), values=values)


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
