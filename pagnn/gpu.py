import logging
import os

import GPUtil
import numba
import numba.cuda

import pagnn

logger = logging.getLogger(__name__)


def test_cuda():
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    numba.cuda.api.detect()
    # The following requires the 'cudatoolkit' package
    numba.cuda.cudadrv.libs.test()


def init_gpu() -> None:
    """Select the least active GPU."""
    if not pagnn.CUDA:
        logger.info("Running on the CPU.")
    else:
        deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, maxMemory=0.5)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in deviceIDs)
        logger.info("Running on GPU number %s.", os.environ['CUDA_VISIBLE_DEVICES'])
