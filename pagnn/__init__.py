"""Top-level package for Protein Adjacency Graph Neural Network.

.. autosummary::
   :toctree: _modules

   pagnn.settings
   pagnn.exc
   pagnn.types
   pagnn.gpu
   pagnn.utils
   pagnn.io
   pagnn.dataset
   pagnn.datapipe
   pagnn.datavardcn
   pagnn.datavargan
   pagnn.models
   pagnn.training
"""
__author__ = """Alexey Strokach"""
__email__ = 'alex.strokach@utoronto.ca'
__version__ = '__version__ = "0.1.13"'

# See: https://github.com/apache/arrow/issues/2637
import pyarrow  # noqa

from . import settings, exc
from .types import *
from .gpu import init_gpu
from .utils import *
from .io import iter_datarows_shuffled, iter_datarows
from .dataset import (row_to_dataset, get_negative_example, get_permuted_examples, get_offset,
                      get_indices)
from . import datapipe, datavardcn, datavargan, models, training
