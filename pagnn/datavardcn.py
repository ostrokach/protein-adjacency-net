"""Prepare data for input into a Dynamic Convolutional Network."""
import logging
from typing import Tuple

import numpy as np
import torch
from torch.autograd import Variable

from pagnn import settings
from pagnn.types import DataSet, DataSetCollection, DataVar, DataVarCollection
from pagnn.utils import expand_adjacency

logger = logging.getLogger(__name__)


def dataset_to_datavar(ds: DataSet, push_seq=True, push_adj=True) -> DataVar:
    """Convert a `DataSet` into a `DataVar`."""
    if push_seq:
        seq = ds.seq.to(settings.device).coalesce().to_dense().unsqueeze(0)
    else:
        seq = None

    if push_adj and ds.adj.nnz != 0:
        adj = expand_adjacency(ds.adj).to(settings.device).to_dense()
    else:
        adj = None

    return DataVar(seq, adj)


def push_dataset_collection(
    dsc: DataSetCollection, push_seq=True, push_adj=True
) -> Tuple[DataVarCollection, Variable]:
    """Convert a `DataSetCollection` into a `DataVarCollection`."""
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
    return (pos, neg), torch.from_numpy(targets).unsqueeze(1)
