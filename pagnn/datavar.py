import logging
from typing import Tuple

import numpy as np
import torch
from scipy import sparse
from torch.autograd import Variable

from . import settings
from .types import DataSet, DataSetCollection, DataVar, DataVarCollection
from .utils import expand_adjacency, get_seq_array

logger = logging.getLogger(__name__)


def to_numpy(array: Variable) -> np.ndarray:
    """Convert PyTorch `Variable` to numpy array."""
    if settings.CUDA:
        return array.data.cpu().numpy()
    else:
        return array.data.numpy()


def to_tensor(array: np.ndarray) -> torch.FloatTensor:
    tensor = torch.FloatTensor(array)
    if settings.CUDA:
        tensor = tensor.cuda()
    return tensor


def to_sparse_tensor(sparray: sparse.spmatrix) -> torch.sparse.FloatTensor:
    i = torch.LongTensor(np.vstack([sparray.row, sparray.col]))
    v = torch.FloatTensor(sparray.data)
    s = torch.Size(sparray.shape)
    tensor = torch.sparse.FloatTensor(i, v, s)
    if settings.CUDA:
        tensor = tensor.cuda()
    return tensor


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


def push_dataset_collection(dsc: DataSetCollection, push_seq=True,
                            push_adj=True) -> Tuple[DataVarCollection, Variable]:
    pos_ds, neg_ds = dsc
    pos = [dataset_to_datavar(ds) for ds in pos_ds]
    neg = [dataset_to_datavar(ds, push_seq, push_adj) for ds in neg_ds]
    # Targets
    targets = [ds.target for ds in pos_ds]
    if push_seq:
        targets += [ds.target for ds in neg_ds if ds.seq is not None]
    if push_adj:
        targets += [ds.target for ds in neg_ds if ds.adj is not None and ds.adj.nnz > 0]
    targets = np.array(targets).astype(np.float64)
    return (pos, neg), Variable(to_tensor(targets).unsqueeze(1))
