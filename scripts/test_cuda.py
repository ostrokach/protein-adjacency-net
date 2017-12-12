import os

import numba
import numba.cuda

os.environ['CUDA_HOME'] = '/usr/local/cuda'

numba.cuda.api.detect()

# The following requires the 'cudatoolkit' package
numba.cuda.cudadrv.libs.test()
