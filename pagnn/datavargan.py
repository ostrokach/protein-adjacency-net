"""Prepare data for input into a Generative Adverserial Network."""
import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from scipy import sparse
from torch.autograd import Variable

from pagnn import settings
from pagnn.dataset import extract_adjacency_from_middle, get_indices
from pagnn.types import DataSetGAN, DataVarGAN
from pagnn.utils import expand_adjacency, get_seq_array, to_numpy, to_sparse_tensor

logger = logging.getLogger(__name__)


def dataset_to_datavar(ds: DataSetGAN) -> DataVarGAN:
    """Convert a `DataSetGAN` into a `DataVarGAN`."""
    ds = pad_edges(ds)
    seqs = push_seqs(ds.seqs)
    adjs = push_adjs(gen_adj_pool(ds.adjs[0]))
    return DataVarGAN(seqs, adjs)


def pad_edges(ds: DataSetGAN,
              target_length=512,
              random_state: Optional[np.random.RandomState] = None):
    """Add padding before and after sequences and adjacency matrix to fit to `target_length`."""
    if random_state is None:
        random_state = np.random.RandomState()

    length = len(ds.seqs[0])

    if length <= target_length:
        new_seqs, new_adjs = _pad_edges_shorter(ds, length, target_length, random_state)
    else:
        new_seqs, new_adjs = _pad_edges_longer(ds, length, target_length, random_state)

    return DataSetGAN(new_seqs, new_adjs, ds.targets, ds.meta)


def _pad_edges_shorter(ds: DataSetGAN, length: int, target_length: int,
                       random_state: np.random.RandomState):
    start = random_state.randint(0, target_length - length + 1)
    pad_end = target_length - length - start

    new_seqs = []
    for seq in ds.seqs:
        new_seq = b'.' * start + seq + b'.' * pad_end
        new_seqs.append(new_seq)

    new_adjs = []
    for adj in ds.adjs:
        row = adj.row + start
        col = adj.col + start
        new_adj = sparse.coo_matrix(
            (adj.data, (row, col)), dtype=adj.dtype, shape=(target_length, target_length))
        new_adjs.append(new_adj)
    return new_seqs, new_adjs


def _pad_edges_longer(ds: DataSetGAN, length: int, target_length: int,
                      random_state: np.random.RandomState):
    start, stop = get_indices(target_length, length, 'middle', random_state)

    new_seqs = []
    for seq in ds.seqs:
        new_seq = seq[start:stop]
        new_seqs.append(new_seq)

    new_adjs = []
    for adj in ds.adjs:
        new_adj = extract_adjacency_from_middle(start, stop, adj)
        new_adjs.append(new_adj)
    return new_seqs, new_adjs


def push_seqs(seqs: List[bytes]) -> Variable:
    """Convert a list of `DataSetGAN` sequences into a `Variable`."""
    seqs = [get_seq_array(seq) for seq in seqs]
    seqs = [to_sparse_tensor(seq) for seq in seqs]
    seqs = [seq.to_dense().unsqueeze(0) for seq in seqs]
    seq_var = Variable(torch.cat(seqs))
    return seq_var


def push_adjs(adjs: List[sparse.spmatrix]) -> Variable:
    """Convert a `DataSetGAN` adjacency into a `Variable`."""
    return [Variable(to_sparse_tensor(expand_adjacency(adj)).to_dense()) for adj in adjs]


def gen_adj_pool(adj: sparse.spmatrix) -> List[sparse.spmatrix]:
    adjs = [adj]
    for i in range(3):
        adjs.append(pool_adjacency_mat(adjs[-1]))
    return adjs


def pool_adjacency_mat(adj: sparse.spmatrix, kernel_size=4, stride=2, padding=1,
                       _mapping_cache={}) -> sparse.spmatrix:
    # assert adj.shape[0] == adj.shape[1]

    if (adj.shape, kernel_size, stride, padding) in _mapping_cache:
        mapping = _mapping_cache[(adj.shape, kernel_size, stride, padding)]
    else:
        mapping = conv2d_mapping(adj.shape[0], kernel_size, stride, padding)
    conv_mat = conv2d_matrix(mapping, adj.row, adj.col, adj.shape, kernel_size, stride, padding)
    return sparse.coo_matrix(conv_mat)


@jit(nopython=True)
def conv2d_matrix(mapping, row, col, shape, kernel_size, stride, padding):
    new_shape = conv2d_shape(shape, kernel_size, stride, padding)
    conv_mat = np.zeros(new_shape, dtype=np.int16)
    for i, (r, c) in enumerate(zip(row, col)):
        conv_mat[slice(*mapping[r]), slice(*mapping[c])] = 1
    return conv_mat


@jit(nopython=True)
def conv2d_shape(shape, kernel_size, stride, padding):
    shape = (
        max(1, math.ceil((shape[0] + 2 * padding - (kernel_size - 1)) / stride)),
        max(1, math.ceil((shape[1] + 2 * padding - (kernel_size - 1)) / stride)),
    )
    return shape


@jit(nopython=True)
def conv2d_mapping(length, kernel_size, stride, padding):
    mapping = [(0, 0) for _ in range(0 - padding, length + padding + kernel_size)]
    for i_conv, start in enumerate(range(0 - padding, length + padding, stride)):
        for i_orig in range(start, start + kernel_size):
            old = mapping[i_orig]
            if old == (0, 0):
                mapping[i_orig] = (i_conv, i_conv + 1)
            else:
                if old[0] > i_conv:
                    mapping[i_orig] = (i_conv, old[1])
                elif old[1] < (i_conv + 1):
                    mapping[i_orig] = (old[0], i_conv + 1)
    return mapping


def pool_adjacency_mat_reference(adj: Variable, kernel_size=4, stride=2, padding=1) -> Variable:
    """Pool and downsample the adjacency matrix `adj` (reference implementation).
    """
    conv_filter = Variable(torch.ones(1, 1, kernel_size, kernel_size))
    if settings.CUDA:
        conv_filter = conv_filter.cuda()
    adj_conv = F.conv2d(adj.unsqueeze(0).unsqueeze(0), conv_filter, stride=stride, padding=padding)
    adj_conv_bool = (adj_conv != 0).float()
    return adj_conv_bool.squeeze()


def pool_adjacency_mat_reference_wrapper(adj: sparse.spmatrix, kernel_size=4, stride=2,
                                         padding=1) -> sparse.spmatrix:
    """Wrapper over `pool_adjacency_mat_reference` to provide the same API as `pool_adjacency_mat`.
    """
    adj = Variable(to_sparse_tensor(adj).to_dense())
    adj_conv = pool_adjacency_mat_reference(adj, kernel_size, stride, padding)
    return sparse.coo_matrix(to_numpy(adj_conv), dtype=np.int16)
