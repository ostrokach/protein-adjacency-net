from typing import List, Tuple

import numpy as np
from scipy import sparse
from torch.nn import Module

from pagnn.datavar import push_dataset_collection
from pagnn.types import DataGen, DataSet
from pagnn.utils import to_numpy

# DataGen = Callable[[], Iterator[DataSetCollection]]


def evaluate_validation_dataset(net: Module,
                                datagen: DataGen,
                                keep_neg_seq: bool = False,
                                keep_neg_adj: bool = False,
                                fake_adj: bool = False):
    """Evaluate the performance of a network on a validation dataset."""
    assert keep_neg_seq or keep_neg_adj
    if fake_adj:
        assert keep_neg_seq and not keep_neg_adj

    outputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for idx, (pos, neg) in enumerate(datagen()):
        if fake_adj:
            pos = [
                DataSet(ds.seq,
                        sparse.coo_matrix((np.ones(len(ds.seq)), (np.arange(len(ds.seq)),
                                                                  np.arange(len(ds.seq))))),
                        ds.target) for ds in pos
            ]
        dvc, targets = push_dataset_collection((pos, neg), keep_neg_seq, keep_neg_adj)
        outputs = net(dvc)
        outputs_list.append(to_numpy(outputs))
        targets_list.append(to_numpy(targets))
    outputs = np.vstack(outputs_list).squeeze()
    targets = np.vstack(targets_list).squeeze()
    assert outputs.ndim == 1
    assert targets.ndim == 1
    return targets, outputs


def evaluate_mutation_dataset(net: Module, datagen: DataGen) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the performance of a network on a mutation dataset.

    Args:
        net: Neural network instance.
        datagen: A collable which returns an iterator over `DataSetCollection` types.

    Returns:
        Targets (true values) and outputs (predicted values) for the given network and datagen.

        For categorical values:
            - `0`: mutation is benign.
            - `-1`: mutation is deleterious.
            - `+1`: not possible since we are assuming that the wild-type protein in stable
                (and cannot be made more stable).

        For continuous values:
            - `0`: mutation has no effect on stability.
            - `-ive`: mutation decreases stability.
            - `+ive`: mutation increases stability.
    """
    outputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for idx, (pos, neg) in enumerate(datagen()):
        dvc, targets = push_dataset_collection((pos, neg), push_seq=True, push_adj=False)
        outputs = net(dvc)
        targets_arr = to_numpy(targets)
        targets_arr_diff = targets_arr[1::2]  # - targets_arr[0::2]
        outputs_arr = to_numpy(outputs)
        outputs_arr_diff = outputs_arr[1::2] - outputs_arr[0::2]
        assert targets_arr_diff.shape == outputs_arr_diff.shape
        targets_list.append(targets_arr_diff)
        outputs_list.append(outputs_arr_diff)
    outputs = np.vstack(outputs_list)
    targets = np.vstack(targets_list)
    return targets, outputs
