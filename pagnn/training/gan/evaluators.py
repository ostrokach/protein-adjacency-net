from typing import List

import numpy as np
import torch

from pagnn.datavargan import dataset_to_datavar
from pagnn.types import DataSetGAN
from pagnn.utils import to_numpy


def evaluate_validation_dataset(net_d, datasets: List[DataSetGAN]):
    """

    Returns:
        A tuple of targets and outputs arrays.
            - Targets are ΔΔG values.
            - Outputs are (pred_mut [low] - pred_wt [high]), so they should be *positive* for
              stabilizing mutations and *negative* for destabilizing mutations (i.e. the
              *reverse* of ΔΔG).
    """
    outputs = []
    targets = []
    for dataset in datasets:
        datavar = dataset_to_datavar(dataset)
        with torch.no_grad():
            output = net_d(*datavar)
        output = to_numpy(output).squeeze()
        target = np.array(dataset.targets)
        outputs.append(output)
        targets.append(target)
    outputs_ar = np.hstack(outputs)
    targets_ar = np.hstack(targets)
    return targets_ar, outputs_ar


def evaluate_mutation_dataset(net_d, datasets: List[DataSetGAN]):
    """

    Returns:
    A tuple of targets and outputs arrays.
        - Targets are 0 for benign, -1 for deleterious.
        - Outputs are (pred_mut [low] - pred_wt [high]), so they should be *positive* for
            stabilizing mutations and *negative* for destabilizing mutations (i.e. the
            *reverse* of ΔΔG).
    """
    outputs = []
    targets = []
    for dataset in datasets:
        datavar = dataset_to_datavar(dataset)
        with torch.no_grad():
            output = net_d(*datavar)
        output = to_numpy(output).squeeze()  # (high, low)
        target = np.array(dataset.targets)  # (1, 0)...
        output = output[1::2] - output[0::2]
        target = target[1::2]
        outputs.append(output)
        targets.append(target)
    outputs_ar = np.hstack(outputs)
    targets_ar = np.hstack(targets)
    return targets_ar, outputs_ar
