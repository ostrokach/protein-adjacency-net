import math

import numpy as np
from numba import jit
from scipy import sparse


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
    # keep_mask = np.ones(len(x.data), dtype=bool)
    # for i, (r, c) in enumerate(zip(x.row, x.col)):
    #     if abs(r - c) <= (bandwidth - 1):
    #         keep_mask[i] = 0
    keep_mask = np.abs(x.row - x.col) >= bandwidth
    x.data = x.data[keep_mask]
    x.row = x.row[keep_mask]
    x.col = x.col[keep_mask]
    return x


def add_eye_sparse(x: sparse.spmatrix, bandwidth: int, copy: bool = True) -> sparse.spmatrix:
    if not bandwidth:
        return x
    if copy:
        x = x.copy()
    data_list = [x.data]
    row_list = [x.row]
    col_list = [x.col]
    # Add diagonal
    data_list.append(np.ones(x.shape[1]))
    row_list.append(np.arange(x.shape[1]))
    col_list.append(np.arange(x.shape[1]))
    # Add off-diagonals
    for k in range(1, bandwidth, 1):
        data_list.extend([np.ones(x.shape[1] - k), np.ones(x.shape[1] - k)])
        row_list.extend([np.arange(x.shape[1] - k), k + np.arange(x.shape[1] - k)])
        col_list.extend([k + np.arange(x.shape[1] - k), np.arange(x.shape[1] - k)])
    x.data = np.hstack(data_list)
    x.row = np.hstack(row_list)
    x.col = np.hstack(col_list)
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
