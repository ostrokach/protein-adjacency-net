"""Data types.

Notes:
  * Not sure if its good practice to have all your types defined in a separate file,
    but it's the only way I've been able to get it to work with mypy.
"""
from typing import Callable, Generator, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import pyarrow as pa
import torch
from scipy import sparse
from torch.autograd import Variable


class SparseMat(NamedTuple):
    indices: torch.LongTensor
    values: torch.FloatTensor
    m: int
    n: int

    def __eq__(self, other):
        equal = (
            np.allclose(self.indices, other.indices)
            and np.allclose(self.values, other.values)
            and self.m == other.m
            and self.n == other.n
        )
        return equal

    def to_sparse_tensor(self):
        tensor = torch.sparse_coo_tensor(self.indices, self.values, size=(self.m, self.n))
        return tensor


class DataRow(NamedTuple):
    """Tuple for storing rows of data from an input table.

    Notes:
        * This data structure can contain other attributes as neccessary.
    """

    sequence: str
    adjacency_idx_1: List[int]
    adjacency_idx_2: List[int]
    target: float
    # ...other columns as neccessary


RowGen = Generator[DataRow, None, None]

RowGenF = Generator[Optional[DataRow], Callable, None]


class DataSet(NamedTuple):
    """The main unit of training / validation data.

    Contains the sequences, the adjacency matric, and the label (target)
    of a protein domain.
    """

    #: Sequence.
    seq: torch.sparse.FloatTensor
    adj: sparse.spmatrix
    target: float
    meta: Optional[dict] = None


class DataSetGAN(NamedTuple):
    """The main storage unit of training / validation data.

    Contains one or more sequences, the adjacency matric, and the label (target)
    of a protein domain.
    """

    #: List of sequences that match a single adjacency.
    seqs: List[SparseMat]
    #: List of adjacencies, ranging from unpooled to 4x pooled.
    adjs: List[SparseMat]
    #: Expected value for every sequence in `seqs`.
    targets: torch.FloatTensor
    #: Optional metadata.
    meta: Optional[dict] = None

    def to_buffer(self):
        data = {
            "seqs": [spm.indices[0, :].numpy().astype(np.uint8) for spm in self.seqs],
            "adj_indices": [spm.indices.numpy().astype(np.uint16) for spm in self.adjs],
            "adj_values": [spm.values.numpy().astype(np.float32) for spm in self.adjs],
            "targets": self.targets.numpy(),
            "meta": self.meta,
        }
        buf = pa.serialize(data).to_buffer()
        return buf

    @classmethod
    def from_buffer(cls, buf):
        data = pa.deserialize(buf)
        seqs = []
        for seq_row in data["seqs"]:
            indices = torch.stack(
                [
                    torch.as_tensor(seq_row, dtype=torch.long),
                    torch.arange(0, len(seq_row), dtype=torch.long),
                ]
            )
            values = torch.ones(len(seq_row), dtype=torch.float)
            seq_length = len(seq_row)
            seqs.append(SparseMat(indices, values, 20, seq_length))
        adjs = []
        for adj_indices, adj_values in zip(data["adj_indices"], data["adj_values"]):
            indices = torch.as_tensor(adj_indices.astype(np.int64), dtype=torch.long)
            values = torch.as_tensor(adj_values, dtype=torch.float)
            adjs.append(SparseMat(indices, values, seq_length, seq_length))
            del seq_length
        targets = torch.from_numpy(data["targets"])
        meta = data["meta"]
        return cls(seqs, adjs, targets, meta)

    def __eq__(self, other):
        if len(self.seqs) != len(other.seqs):
            return False

        if not all([sp1 == sp2 for sp1, sp2 in zip(self.seqs, other.seqs)]):
            return False

        if len(self.adjs) != len(other.adjs):
            return False

        if not all([sp1 == sp2 for sp1, sp2 in zip(self.adjs, other.adjs)]):
            return False

        if not np.allclose(self.targets, other.targets):
            return False

        if self.meta != other.meta:
            return False

        return True


DataSetGenM = Generator[Optional[DataSetGAN], DataSetGAN, None]


class DataVar(NamedTuple):
    """Input variables for the Deep Convolutional Network."""

    seq: Variable
    adj: Variable


class DataVarGAN(NamedTuple):
    """Input variables for the Generative Adverserial Network."""

    seqs: torch.FloatTensor
    adjs: torch.sparse.FloatTensor


DataSetCollection = Tuple[List[DataSet], List[DataSet]]
"""A collection of +ive and -ive training examples."""

DataGen = Callable[[], Iterator[DataSetCollection]]
"""A function which returns an iterator over dataset collections."""

DataVarCollection = Tuple[List[DataVar], List[DataVar]]
"""A collection of +ive and -ive training examples."""
