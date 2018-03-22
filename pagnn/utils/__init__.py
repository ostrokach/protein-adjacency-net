"""Broadly-applicable helper functions.

.. autosummary::
   :toctree: _modules

   argmax_onehot
   to_numpy
   to_tensor
   to_sparse_tensor
   seq_to_array
   array_to_seq
   get_adjacency
   expand_adjacency
   get_seq_identity
   get_adj_identity
   iter_forever
   iter_submodules
   make_weblogo
   score_blosum62
   score_edit
   get_version
"""
from .array_ops import argmax_onehot, to_numpy, to_sparse_tensor, to_tensor
from .dataset_ops import (expand_adjacency, get_adj_identity, get_adjacency,
                          seq_to_array, array_to_seq, get_seq_identity, AMINO_ACIDS)
from .interpolation import interpolate_adjacencies, interpolate_sequences
from .iter_ops import iter_forever, iter_submodules
from .weblogo import make_weblogo
from .tensorboard import add_image
from .scoring import score_blosum62, score_edit
from .training import get_version
