from typing import List, NamedTuple, Optional

from scipy import sparse
from torch.autograd import Variable


class DataSet(NamedTuple):
    """The main storage unit of training / validation data.

    Contains one or more sequences, the adjacency matric, and the label (target)
    of a protein domain.
    """
    seqs: List[bytes]
    adj: sparse.spmatrix
    target: float
    meta: Optional[dict] = None


class DataVar(NamedTuple):
    """Input variables for the neural network."""
    seqs: Variable
    adjs: List[Variable]
