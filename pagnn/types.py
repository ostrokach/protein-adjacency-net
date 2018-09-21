"""Data types.

Notes:
  * Not sure if its good practice to have all your types defined in a separate file,
    but it's the only way I've been able to get it to work with mypy.
"""
from typing import Callable, Generator, Iterator, List, NamedTuple, Optional, Tuple

from scipy import sparse
from torch.autograd import Variable


class DataRow(NamedTuple):
    """Tuple for storing rows of data from an input table.

    Notes:
        * This data structure can contain other attributes as neccessary.
    """

    sequence: str
    adjacency_idx_1: List[int]
    adjacency_idx_2: List[int]
    # ...other columns as neccessary


RowGen = Generator[DataRow, None, None]

RowGenF = Generator[Optional[DataRow], Callable, None]


class DataSet(NamedTuple):
    """The main unit of training / validation data.

    Contains the sequences, the adjacency matric, and the label (target)
    of a protein domain.
    """

    #: Sequence.
    seq: bytes
    adj: sparse.spmatrix
    target: float
    meta: Optional[dict] = None


class DataSetGAN(NamedTuple):
    """The main storage unit of training / validation data.

    Contains one or more sequences, the adjacency matric, and the label (target)
    of a protein domain.
    """

    #: List of sequences that match a single adjacency.
    seqs: List[bytes]
    #: List of adjacencies, ranging from unpooled to 4x pooled.
    adjs: sparse.spmatrix
    #: Expected value for every sequence in `seqs`.
    targets: List[float]
    #: Optional metadata.
    meta: Optional[dict] = None


DataSetGenM = Generator[Optional[DataSetGAN], DataSetGAN, None]


class DataVar(NamedTuple):
    """Input variables for the Deep Convolutional Network."""

    seq: Variable
    adj: Variable


class DataVarGAN(NamedTuple):
    """Input variables for the Generative Adverserial Network."""

    seqs: Variable
    adjs: Variable


DataSetCollection = Tuple[List[DataSet], List[DataSet]]
"""A collection of +ive and -ive training examples."""

DataGen = Callable[[], Iterator[DataSetCollection]]
"""A function which returns an iterator over dataset collections."""

DataVarCollection = Tuple[List[DataVar], List[DataVar]]
"""A collection of +ive and -ive training examples."""
