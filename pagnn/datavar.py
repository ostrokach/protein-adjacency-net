import logging
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
from scipy import sparse
from torch.autograd import Variable

import pagnn
from pagnn import expand_adjacency, get_seq_array

from .dataset import DataSet, DataSetCollection

logger = logging.getLogger(__name__)


class DataVar(NamedTuple):
    seq: Variable
    adj: Variable


DataVarCollection = Tuple[List[DataVar], List[DataVar]]
"""A collection of +ive and -ive training examples."""


def to_numpy(array: Variable) -> np.ndarray:
    """Convert PyTorch `Variable` to numpy array."""
    if pagnn.CUDA:
        return array.data.cpu().numpy()
    else:
        return array.data.numpy()


def to_tensor(array: np.ndarray) -> torch.FloatTensor:
    tensor = torch.FloatTensor(array)
    if pagnn.CUDA:
        tensor = tensor.cuda()
    return tensor


def to_sparse_tensor(sparray: sparse.spmatrix) -> torch.sparse.FloatTensor:
    i = torch.LongTensor(np.vstack([sparray.row, sparray.col]))
    v = torch.FloatTensor(sparray.data)
    s = torch.Size(sparray.shape)
    tensor = torch.sparse.FloatTensor(i, v, s)
    if pagnn.CUDA:
        tensor = tensor.cuda()
    return tensor


# def push_array(array: np.ndarray) -> Variable:
#     """Convert numpy array to PyTorch `Variable`."""
#     tensor = torch.Tensor(array)
#     if pagnn.CUDA:
#         tensor = tensor.cuda()
#     return Variable(tensor)

# def push_sparse_array(sparray: sparse.spmatrix) -> Variable:
#     tensor = to_sparse_tensor(sparray)
#     if pagnn.CUDA:
#         tensor = tensor.cuda()
#     return Variable(tensor)


def dataset_to_datavar(ds: DataSet, push_seq=True, push_adj=True) -> DataVar:
    if push_seq:
        seq = to_sparse_tensor(get_seq_array(ds.seq))
        seq = Variable(seq.to_dense().unsqueeze(0))
    else:
        seq = None

    if push_adj and ds.adj.nnz != 0:
        adj = to_sparse_tensor(expand_adjacency(ds.adj))
        adj = Variable(adj.to_dense())
    else:
        adj = None

    return DataVar(seq, adj)


def push_dataset_collection(dsc: DataSetCollection, keep_neg_seq=True,
                            keep_neg_adj=True) -> DataVarCollection:
    pos_ds, neg_ds = dsc
    pos = [dataset_to_datavar(ds) for ds in pos_ds]
    neg = [dataset_to_datavar(ds, keep_neg_seq, keep_neg_adj) for ds in neg_ds]
    return pos, neg


def get_training_targets(dvc: DataVarCollection) -> Variable:
    pos, neg = dvc
    num_neg_seq = sum(1 for seq, _ in neg if seq is not None)
    num_neg_adj = sum(1 for _, adj in neg if adj is not None)
    return Variable(to_tensor([1] + [0] * (num_neg_seq + num_neg_adj)).unsqueeze(1))
