"""Broadly-applicable helper functions.

.. autosummary::
   :toctree: _modules

   ArgsBase
   argmax_onehot
   conv1d_shape
   conv1d_shape_ceil
   conv2d_shape
   to_numpy
   to_tensor
   to_sparse_tensor
   reshape_internal_dim
   unfold_to
   unfold_from
   padding_amount
   remove_eye
   remove_eye_sparse
   add_eye_sparse
   seq_to_array
   array_to_seq
   get_adjacency
   expand_adjacency
   get_seq_identity
   get_adj_identity
   AMINO_ACIDS
   eval_net
   freeze_net
   unfreeze_net
   freeze_adj_conv
   unfreeze_adj_conv
   interpolate_adjacencies
   interpolate_sequences
   iter_forever
   iter_submodules
   make_weblogo
   add_image
   score_blosum62
   score_edit
   random_sequence
   set_cuda

.. autosummary::
    :toctree: _modules

    str_to_path
    load_yaml
    dump_yaml
"""
from .args import ArgsBase
from .array_ops import (argmax_onehot, conv1d_shape, conv1d_shape_ceil, conv2d_shape, to_numpy,
                        to_sparse_tensor, to_tensor, reshape_internal_dim, unfold_to, unfold_from,
                        padding_amount, remove_eye, remove_eye_sparse, add_eye_sparse)
from .dataset_ops import (expand_adjacency, get_adj_identity, get_adjacency, seq_to_array,
                          array_to_seq, get_seq_identity, AMINO_ACIDS)
from .datetime_ops import convert_to_timedelta
from .network_ops import eval_net, freeze_net, unfreeze_net, freeze_adj_conv, unfreeze_adj_conv
from .interpolation import interpolate_adjacencies, interpolate_sequences
from .iter_ops import iter_forever, iter_submodules
from .weblogo import make_weblogo
from .tensorboard import add_image
from .scoring import score_blosum62, score_edit
from .testing import random_sequence, set_cuda
from .py_ops import str_to_path, load_yaml, dump_yaml
