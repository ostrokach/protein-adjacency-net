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

    TODO: seqs, adjs, and targets should all be tensors.
    """

    #: List of sequences that match a single adjacency.
    seqs: List[torch.sparse.FloatTensor]
    #: List of adjacencies, ranging from unpooled to 4x pooled.
    adjs: List[sparse.spmatrix]
    #: Expected value for every sequence in `seqs`.
    targets: torch.FloatTensor
    #: Optional metadata.
    meta: Optional[dict] = None

    def to_buffer(self):
        data = {
            "seqs": [seq._indices()[0, :].numpy() for seq in self.seqs],
            "adjs": [(adj.row, adj.col) for adj in self.adjs],
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
            index = torch.stack(
                [torch.from_numpy(seq_row), torch.arange(0, len(seq_row), dtype=torch.long)]
            )
            values = torch.ones(len(seq_row), dtype=torch.float)
            size = (20, len(seq_row))
            seq = torch.sparse_coo_tensor(index, values, size=size)
            seqs.append(seq)
        adjs = []
        for row, col in data["adjs"]:
            values = np.ones(len(col))
            seq_size = len(data["seqs"][0])
            adj = sparse.coo_matrix((values, (row, col)), shape=(seq_size, seq_size))
            adjs.append(adj)
        targets = torch.from_numpy(data["targets"])
        meta = data["meta"]
        return cls(seqs, adjs, targets, meta)

    def __eq__(self, other):
        if len(self.seqs) != len(other.seqs):
            return False
        if not all(
            [
                np.allclose(self.seqs[i]._indices(), other.seqs[i]._indices())
                for i in range(len(self.seqs))
            ]
        ):
            return False
        if not all(
            [
                np.allclose(self.seqs[i]._values(), other.seqs[i]._values())
                for i in range(len(self.seqs))
            ]
        ):
            return False

        if len(self.adjs) != len(other.adjs):
            return False
        if not all(
            [np.allclose(self.adjs[i].row, other.adjs[i].row) for i in range(len(self.adjs))]
        ):
            return False
        if not all(
            [np.allclose(self.adjs[i].col, other.adjs[i].col) for i in range(len(self.adjs))]
        ):
            return False
        if not all(
            [np.allclose(self.adjs[i].data, other.adjs[i].data) for i in range(len(self.adjs))]
        ):
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

    seqs: Variable
    adjs: Variable


DataSetCollection = Tuple[List[DataSet], List[DataSet]]
"""A collection of +ive and -ive training examples."""

DataGen = Callable[[], Iterator[DataSetCollection]]
"""A function which returns an iterator over dataset collections."""

DataVarCollection = Tuple[List[DataVar], List[DataVar]]
"""A collection of +ive and -ive training examples."""
