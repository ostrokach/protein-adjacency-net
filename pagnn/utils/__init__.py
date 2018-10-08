"""Broadly-applicable helper functions.

.. autosummary::
   :toctree: _modules

   ArgsBase
   argmax_onehot
   conv1d_shape
   conv1d_shape_ceil
   conv2d_shape
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
   score_blosum62
   score_edit
   random_sequence

.. autosummary::
    :caption: py_ops
    :toctree: _modules

    str_to_path
    load_yaml
    dump_yaml
"""
# No interdependencies
from .array_ops import (
    add_eye_sparse,
    conv1d_shape,
    conv1d_shape_ceil,
    conv2d_shape,
    padding_amount,
    remove_eye,
    remove_eye_sparse,
    reshape_internal_dim,
    unfold_from,
    unfold_to,
)
from .converters import *
from .dataset_ops import (
    AMINO_ACIDS,
    array_to_seq,
    expand_adjacency,
    get_adj_identity,
    get_adjacency,
    get_seq_identity,
    seq_to_array,
)
from .stats import StatsBase
from .tensor_ops import (
    argmax_onehot,
    to_sparse_tensor,
)
from .interpolation import interpolate_adjacencies, interpolate_sequences
from .iter_ops import iter_forever, iter_submodules
from .network_ops import eval_net, freeze_adj_conv, freeze_net, unfreeze_adj_conv, unfreeze_net
from .testing import random_sequence, set_device

# Other
from .args import ArgsBase
from .checkpoint import load_checkpoint, validate_checkpoint, write_checkpoint
from .evaluators import evaluate_validation_dataset, evaluate_mutation_dataset
from .generators import (
    basic_permuted_sequence_adder,
    get_rowgen_mut,
    buffered_permuted_sequence_adder,
    negative_sequence_adder,
)
from .scoring import score_blosum62, score_edit
