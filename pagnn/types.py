from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple

from scipy import sparse
from torch.autograd import Variable


class DataRow(NamedTuple):
    sequence: str
    adjacency_idx_1: List[int]
    adjacency_idx_2: List[int]
    # ...other columns as neccessary


class DataSet(NamedTuple):
    seq: bytes
    adj: sparse.spmatrix
    target: float
    meta: Optional[dict] = None


class DataVar(NamedTuple):
    seq: Variable
    adj: Variable


DataSetCollection = Tuple[List[DataSet], List[DataSet]]
"""A collection of +ive and -ive training examples."""

DataGen = Callable[[], Iterator[DataSetCollection]]
"""A function which returns an iterator over dataset collections."""

DataVarCollection = Tuple[List[DataVar], List[DataVar]]
"""A collection of +ive and -ive training examples."""
