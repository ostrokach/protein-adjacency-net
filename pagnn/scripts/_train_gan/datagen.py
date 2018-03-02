from typing import Generator, Iterator

import numpy as np

from pagnn.dataset import get_negative_example, get_offset
from pagnn.types import DataRow, DataSet, DataSetGAN


def add_permuted_sequence(rowgen: Iterator[DataRow], num_sequences: int,
                          random_state) -> Generator[DataSetGAN, DataSetGAN, None]:
    """

    Args:
        rowgen: Used for **pre-populating** the generator only!
        num_sequences: Number of sequences to generate in each iteration.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        seq = dsg.seqs[0]
        negative_seqs = []
        for _ in range(num_sequences):
            offset = get_offset(len(seq), random_state)
            negative_seq = seq[offset:] + seq[:offset]
            negative_seqs.append(negative_seq)
        negative_dsg = dsg._replace(seqs=negative_seqs)


def add_negative_sequences(rowgen: Iterator[DataRow], method: str, num_sequences: int,
                           random_state) -> Generator[DataSetGAN, DataSetGAN, None]:
    """

    Args:
        rowgen: Generator used for fetching negative rows.
        method: Method by which longer negative sequences get cut to the correct size.
        num_sequences: Number of sequences to generate in each iteration.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        ds = DataSet(dsg.seqs[0], dsg.adjs[0], dsg.targets[0])
        negative_seqs = []
        for _ in range(num_sequences):
            negative_ds = get_negative_example(ds, method, rowgen, random_state)
            negative_seqs.append(negative_ds.seq)
        negative_dsg = dsg._replace(seqs=negative_seqs)
