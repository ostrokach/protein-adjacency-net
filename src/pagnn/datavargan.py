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
from pagnn.types import DataSetGAN, DataVarGAN, SparseMat
from pagnn.utils import add_eye_sparse, conv2d_shape, remove_eye_sparse, to_sparse_tensor

logger = logging.getLogger(__name__)


def datasets_to_datavar(dss: List[DataSetGAN]) -> DataVarGAN:
    """Convert a list of `DataSetGAN` into a `DataVarGAN`."""
    seqs = push_seqs([b"".join(s) for s in zip(*[ds.seqs for ds in dss])])
    adjs = [push_adjs(gen_adj_pool(ds.adjs[0])) for ds in dss]
    return DataVarGAN(seqs, adjs)


def dataset_to_datavar(
    ds: DataSetGAN,
    n_convs: int,
    kernel_size: int,
    stride: int,
    padding: int,
    remove_diags: int,
    add_diags: bool,
) -> DataVarGAN:
    """Convert a `DataSetGAN` into a `DataVarGAN`."""
    # ds = pad_edges(ds, offset=offset)
    seqs = push_seqs(ds.seqs)
    adj = ds.adjs[0]
    # adj = remove_eye_sparse(adj, remove_diags)
    # adj = add_eye_sparse(adj, add_diags)
    if n_convs <= 1:
        adj_pool = [adj]
    else:
        adj_pool = gen_adj_pool(
            adj,
            n_convs=n_convs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            remove_diags=remove_diags,
            add_diags=add_diags,
        )
    adjs = push_adjs(adj_pool)
    return DataVarGAN(seqs, adjs)


def pad_edges(
    ds: DataSetGAN,
    random_state: Optional[np.random.RandomState] = None,
    target_length: Optional[int] = None,
    offset: Optional[int] = None,
):
    """Add padding before and after sequences and adjacency matrix to fit to `target_length`."""
    if random_state is None:
        random_state = np.random.RandomState()

    length = len(ds.seqs[0])
    if target_length is None:
        target_length = math.ceil(length / 128) * 128

    if length <= target_length or offset is not None:
        new_seqs, new_adjs = _pad_edges_shorter(ds, length, target_length, random_state, offset)
    else:
        new_seqs, new_adjs = _pad_edges_longer(ds, length, target_length, random_state)

    return DataSetGAN(new_seqs, new_adjs, ds.targets, ds.meta)


def _pad_edges_shorter(
    ds: DataSetGAN,
    length: int,
    target_length: int,
    random_state: np.random.RandomState,
    offset: Optional[int] = None,
):
    if offset is None:
        start = random_state.randint(0, target_length - length + 1)
    else:
        start = offset
    pad_end = max(0, target_length - length - start)

    new_seqs = []
    for seq in ds.seqs:
        new_seq = b"." * start + seq + b"." * pad_end
        new_seqs.append(new_seq)

    new_adjs = []
    for adj in ds.adjs:
        row = adj.row + start
        col = adj.col + start
        new_adj = sparse.coo_matrix(
            (adj.data, (row, col)), dtype=adj.dtype, shape=(target_length, target_length)
        )
        new_adjs.append(new_adj)
    return new_seqs, new_adjs


def _pad_edges_longer(
    ds: DataSetGAN, length: int, target_length: int, random_state: np.random.RandomState
):
    start, stop = get_indices(target_length, length, "middle", random_state)

    new_seqs = []
    for seq in ds.seqs:
        new_seq = seq[start:stop]
        new_seqs.append(new_seq)

    new_adjs = []
    for adj in ds.adjs:
        new_adj = extract_adjacency_from_middle(start, stop, adj)
        new_adjs.append(new_adj)
    return new_seqs, new_adjs


def push_seqs(seqs: List[SparseMat]) -> torch.FloatTensor:
    seqs_ts = [seq.to_sparse_tensor().to(settings.device).to_dense().unsqueeze(0) for seq in seqs]
    seq_t = torch.cat(seqs_ts)
    return seq_t


def push_adjs(adjs: List[SparseMat]) -> torch.FloatTensor:
    adjs = [adj.to_sparse_tensor().to(settings.device) for adj in adjs]
    return adjs


def gen_adj_pool(
    adj: SparseMat,
    n_convs: int,
    kernel_size: int,
    stride: int,
    padding: int,
    remove_diags: int,
    add_diags: int,
) -> List[SparseMat]:
    adjs = [adj]
    for i in range(n_convs):
        adj = pool_adjacency_mat(adjs[-1], kernel_size=kernel_size, stride=stride, padding=padding)
        adj = remove_eye_sparse(adj, remove_diags)
        adj = add_eye_sparse(adj, add_diags)
        adjs.append(adj)
    return adjs


def pool_adjacency_mat(
    adj: SparseMat, kernel_size=4, stride=2, padding=1, _mapping_cache={}
) -> SparseMat:
    if (adj.n, kernel_size, stride, padding) in _mapping_cache:
        mapping = _mapping_cache[(adj.n, kernel_size, stride, padding)]
    else:
        mapping = conv2d_mapping(adj.n, kernel_size, stride, padding)
        _mapping_cache[(adj.n, kernel_size, stride, padding)] = mapping
    conv_mat = conv2d_matrix(
        mapping,
        adj.indices[0, :].numpy(),
        adj.indices[1, :].numpy(),
        (adj.m, adj.n),
        kernel_size,
        stride,
        padding,
    )
    sp = sparse.coo_matrix(conv_mat)
    indices = torch.stack(
        [torch.tensor(sp.row, dtype=torch.long), torch.tensor(sp.col, dtype=torch.long)]
    )
    values = torch.tensor(sp.data, dtype=torch.float)
    return SparseMat(indices, values, sp.shape[0], sp.shape[1])


@jit(nopython=True)
def conv2d_matrix(mapping, row, col, shape, kernel_size, stride, padding):
    new_shape = conv2d_shape(shape, kernel_size, stride, padding)
    conv_mat = np.zeros(new_shape, dtype=np.int16)
    for i, (r, c) in enumerate(zip(row, col)):
        conv_mat[slice(*mapping[r]), slice(*mapping[c])] = 1
    return conv_mat


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
    """Pool and downsample the adjacency matrix `adj` (reference implementation)."""
    conv_filter = torch.ones(
        1, 1, kernel_size, kernel_size, device=settings.device, requires_grad=True
    )
    adj_conv = F.conv2d(adj.unsqueeze(0).unsqueeze(0), conv_filter, stride=stride, padding=padding)
    adj_conv_bool = (adj_conv != 0).float()
    return adj_conv_bool.squeeze()


def pool_adjacency_mat_reference_wrapper(
    adj: sparse.spmatrix, kernel_size=4, stride=2, padding=1
) -> sparse.spmatrix:
    """Wraps `pool_adjacency_mat_reference` to provide the same API as `pool_adjacency_mat`"""
    adj = Variable(to_sparse_tensor(adj).to_dense())
    adj_conv = pool_adjacency_mat_reference(adj, kernel_size, stride, padding)
    return sparse.coo_matrix(adj_conv.data.numpy(), dtype=np.int16)
