"""
Not sure if its good practice to have all your types defined in a separate file,
but it's the only way I've been able to get it to work with mypy.
"""
from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple

from scipy import sparse
from torch.autograd import Variable


class DataRow(NamedTuple):
    """Tuple for storing rows of data from an input table."""
    sequence: str
    adjacency_idx_1: List[int]
    adjacency_idx_2: List[int]
    # ...other columns as neccessary


class DataSet(NamedTuple):
    """The main unit of training / validation data.

    Contains the sequences, the adjacency matric, and the label (target)
    of a protein domain.
    """
    seq: bytes
    adj: sparse.spmatrix
    target: float
    meta: Optional[dict] = None


class DataVar(NamedTuple):
    """Input to the neural network."""
    seq: Variable
    adj: Variable


DataSetCollection = Tuple[List[DataSet], List[DataSet]]
"""A collection of +ive and -ive training examples."""

DataGen = Callable[[], Iterator[DataSetCollection]]
"""A function which returns an iterator over dataset collections."""

DataVarCollection = Tuple[List[DataVar], List[DataVar]]
"""A collection of +ive and -ive training examples."""
