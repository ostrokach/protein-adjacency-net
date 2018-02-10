import logging
import os

import GPUtil
import numba
import numba.cuda

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
        device_ids = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, maxMemory=0.5)
        device_id = ','.join(str(i) for i in device_ids)
    else:
        device_id = str(gpu_idx)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    logger.info("Running on GPU number %s.", os.environ['CUDA_VISIBLE_DEVICES'])
