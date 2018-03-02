"""Broadly-applicable helper functions.

.. autosummary::
   :toctree: _modules

   to_numpy
   to_tensor
   to_sparse_tensor
   get_seq_array
   get_adjacency
   expand_adjacency
   get_seq_identity
   get_adj_identity
   iter_forever
   iter_submodules
"""
from .array_ops import to_numpy, to_sparse_tensor, to_tensor
from .dataset_ops import (expand_adjacency, get_adj_identity, get_adjacency,
                          get_seq_array, get_seq_identity)
from .interpolation import interpolate_adjacencies, interpolate_sequences
from .iter_ops import iter_forever, iter_submodules
