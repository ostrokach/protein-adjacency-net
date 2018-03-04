"""Prepare data for input into a Generative Adverserial Network."""
import logging
from typing import List, Optional

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from scipy import sparse
from torch.autograd import Variable

from pagnn.types import DataSetGAN, DataVarGAN
from pagnn.utils import expand_adjacency, get_seq_array, to_sparse_tensor

logger = logging.getLogger(__name__)


def dataset_to_datavar(ds: DataSetGAN, push_seq: bool = True, push_adj: bool = True) -> DataVarGAN:
    """Convert a `DataSetGAN` into a `DataVarGAN`."""
    ds = pad_edges(ds)
    seqs = _push_seqs(ds.seqs)
    adjs = _push_adjs(_gen_adj_pool(ds.adjs[0]))
    return DataVarGAN(seqs, adjs)


def pad_edges(ds: DataSetGAN,
              target_length=512,
              random_state: Optional[np.random.RandomState] = None):
    """Add padding before and after sequences and adjacency matrix to fit to `target_lenght`."""
    if random_state is None:
        random_state = np.random.RandomState()

    length = len(ds.seqs[0])
    start = random_state.randint(0, target_length - length + 1)
    stop = target_length - length - start

    new_seqs = []
    for seq in ds.seqs:
        new_seq = b'.' * start + seq + b'.' * stop
        new_seqs.append(new_seq)

    new_adjs = []
    for adj in ds.adjs:
        row = adj.row + start
        col = adj.col + start
        new_adj = sparse.coo_matrix(
            (adj.data, (row, col)), dtype=adj.dtype, shape=(target_length, target_length))
        new_adjs.append(new_adj)

    new_ds = DataSetGAN(new_seqs, new_adjs, ds.targets, ds.meta)
    return new_ds


def pool_adjacency_mat(adj: sparse.spmatrix, kernel_size=5, stride=2, padding=2) -> sparse.spmatrix:
    """Pool and downsample the adjacency matrix `adj`."""
    row = []
    col = []
    # Go over rows
    for i, i_start in enumerate(range(-padding, adj.shape[0] + padding - kernel_size, stride)):
        i_end = i_start + kernel_size
        # Go over columns
        for j, j_start in enumerate(range(-padding, adj.shape[0] + padding - kernel_size, stride)):
            j_end = j_start + kernel_size
            # Place a value whenever there is at least one non-zero element
            if np.any((adj.row >= i_start) & (adj.row < i_end) & (adj.col >= j_start) &
                      (adj.col < j_end)):
                row.append(i)
                col.append(j)

    # TODO(AS): This won't always work.
    shape = (adj.shape[0] // stride, adj.shape[1] // stride)

    adj_conv = sp.sparse.coo_matrix(
        (np.ones(len(row)), (np.array(row), np.array(col))), shape=shape, dtype=np.int16)
    return adj_conv


def pool_adjacency_mat_reference(adj: np.ndarray, kernel_size=5, stride=2, padding=2) -> np.ndarray:
    """Pool and downsample the adjacency matrix `adj` (reference implementation)."""
    adj_conv = F.conv2d(
        Variable(torch.eye(10).unsqueeze(0).unsqueeze(0)),
        Variable(torch.ones(1, 1, 5, 5)),
        stride=2,
        padding=2)
    adj_conv_bool = (adj_conv != 0).astype(np.int16)
    return adj_conv_bool.squeeze().data.numpy()


def _push_seqs(seqs: List[bytes]) -> Variable:
    """Convert a list of `DataSetGAN` sequences into a `Variable`."""
    seq_tensors = [to_sparse_tensor(get_seq_array(seq)) for seq in seqs]
    seq_tensors_dense = [seq.to_dense().unsqueeze(0) for seq in seq_tensors]
    seq_var = Variable(torch.cat(seq_tensors_dense))
    return seq_var


def _push_adjs(adjs: List[sparse.spmatrix]) -> Variable:
    """Convert a `DataSetGAN` adjacency into a `Variable`."""
    adj_vars = []
    for adj in adjs:
        adj = to_sparse_tensor(expand_adjacency(adj))
        adj = Variable(adj.to_dense())
        adj_vars.append(adj)

    return adj_vars


def _gen_adj_pool(adj: sparse.spmatrix) -> List[sparse.spmatrix]:
    adjs = [adj]
    for i in range(3):
        adjs.append(pool_adjacency_mat(adjs[-1]))
    return adjs
