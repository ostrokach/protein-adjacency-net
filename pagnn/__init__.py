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
   pagnn.datavardcn
   pagnn.datavargan
   pagnn.models
   pagnn.training
"""
__author__ = """Alexey Strokach"""
__email__ = 'alex.strokach@utoronto.ca'
__version__ = '0.1.9.dev'

from . import settings, exc
from .types import *
from .gpu import init_gpu
from .utils import *
from .io import iter_datarows_shuffled, iter_datarows, get_folder_weights
from .dataset import (row_to_dataset, get_negative_example, get_permuted_examples, get_offset,
                      get_indices)
from . import datavardcn, datavargan, models, training
