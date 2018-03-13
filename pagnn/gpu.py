"""GPU configuration."""
import io
import logging
import os
import shlex
import subprocess
from typing import List

import numba
import numba.cuda
import pandas as pd

from . import settings

logger = logging.getLogger(__name__)


def test_cuda():
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    numba.cuda.api.detect()
    # The following requires the 'cudatoolkit' package
    numba.cuda.cudadrv.libs.test()


def init_gpu(gpu_idx: int = None) -> None:
    """Specify which GPU should be used (or select the least active one)."""
    assert settings.CUDA
    if gpu_idx is None:
        device_ids = get_available_gpus(max_load=0.5, max_memory=0.5)
        device_id = ','.join(str(i) for i in device_ids)
    else:
        device_id = str(gpu_idx)
    # TODO: This does not seem to work...
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    logger.info("Running on GPU number %s.", os.environ['CUDA_VISIBLE_DEVICES'])


def get_available_gpus(max_load: float = 0.5, max_memory: float = 0.5) -> List[int]:
    """Get a list of GPUs with load under the given load requirements.

    Args:
        max_load: Max fraction of GPU cycles used.
        max_memory: Max fractio of GPU memory used.

    Returns:
        A list of GPU ids for GPUs which meet the load requirements.
    """
    assert 0 <= max_load <= 1
    assert 0 <= max_memory <= 1

    columns = [
        'index', 'utilization.gpu', 'memory.total', 'memory.used', 'memory.free', 'driver_version',
        'name', 'gpu_serial', 'display_active', 'display_mode'
    ]

    system_command = f"nvidia-smi --query-gpu={','.join(columns)} --format=csv,noheader,nounits"

    proc = subprocess.run(
        shlex.split(system_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)

    buf = io.StringIO()
    buf.write(proc.stdout)
    buf.seek(0)

    df = pd.read_csv(buf, names=columns)
    df['memory_utilization'] = df['memory.used'] / df['memory.total']
    df = df \
        .sort_values(['memory.free'], ascending=False) \
        .sort_values(['utilization.gpu'], ascending=True)
    df = df[((df['utilization.gpu'] / 100) <= max_load) & (df['memory_utilization'] <= max_memory)]
    return df['index'].tolist()
