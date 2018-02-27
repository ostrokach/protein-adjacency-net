import logging
from typing import List, Optional

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from scipy import sparse
from torch.autograd import Variable

from .datavar import to_sparse_tensor
from .types_gan import DataSet, DataVar
from .utils import expand_adjacency, get_seq_array

logger = logging.getLogger(__name__)


def dataset_to_datavar(ds: DataSet, push_seq=True, push_adj=True) -> DataVar:
    ds = pad_edges(ds)
    seqs = push_seqs(ds.seqs)
    adjs = push_adjs(gen_adj_pool(ds.adj))
    return DataVar(seqs, adjs)


def pad_edges(ds: DataSet, target_length=512, random_state: Optional[np.random.RandomState] = None):
    if random_state is None:
        random_state = np.random.RandomState()

    length = len(ds.seqs[0])
    start = random_state.randint(0, target_length - length + 1)
    stop = target_length - length - start

    new_seqs = []
    for seq in ds.seqs:
        new_seq = b'.' * start + seq + b'.' * stop
        new_seqs.append(new_seq)

    row = ds.adj.row + start
    col = ds.adj.col + start
    new_adj = sparse.coo_matrix(
        (ds.adj.data, (row, col)), dtype=ds.adj.dtype, shape=(target_length, target_length))

    new_ds = DataSet(new_seqs, new_adj, ds.target, ds.meta)
    return new_ds


def push_seqs(seqs: List[bytes]) -> Variable:
    seq_tensors = [to_sparse_tensor(get_seq_array(seq)) for seq in seqs]
    seq_tensors_dense = [seq.to_dense().unsqueeze(0) for seq in seq_tensors]
    seq_var = Variable(torch.cat(seq_tensors_dense))
    return seq_var


def push_adjs(adjs: List[sparse.spmatrix]) -> Variable:
    adj_vars = []
    for adj in adjs:
        adj = to_sparse_tensor(expand_adjacency(adj))
        adj = Variable(adj.to_dense())
        adj_vars.append(adj)

    return adj_vars


def gen_adj_pool(adj: sparse.spmatrix) -> List[sparse.spmatrix]:
    adjs = [adj]
    for i in range(3):
        adjs.append(pool_adjacency_mat(adjs[-1]))
    return adjs


def pool_adjacency_mat(adj: sparse.spmatrix, kernel_size=5, stride=2, padding=2) -> sparse.spmatrix:
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
    adj_conv = F.conv2d(
        Variable(torch.eye(10).unsqueeze(0).unsqueeze(0)),
        Variable(torch.ones(1, 1, 5, 5)),
        stride=2,
        padding=2)
    adj_conv_bool = (adj_conv != 0).astype(np.int16)
    return adj_conv_bool.squeeze().data.numpy()
