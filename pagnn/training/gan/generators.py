import itertools
import logging
from typing import Callable, Generator, Optional, Tuple

import numpy as np

from pagnn import exc
from pagnn.dataset import get_negative_example, row_to_dataset
from pagnn.types import DataRow, DataSet, DataSetGAN

logger = logging.getLogger(__name__)

RowGen = Generator[DataRow, Tuple[Callable, int], None]


def basic_permuted_sequence_adder(num_sequences: int,
                                  keep_pos: bool,
                                  random_state: Optional[np.random.RandomState] = None):

    if random_state is None:
        random_state = np.random.RandomState()
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        seq = dsg.seqs[0]
        negative_seqs = []
        for _ in range(num_sequences):
            offset = random_state.randint(0, len(seq))
            negative_seq = seq[offset:] + seq[:offset]
            negative_seqs.append(negative_seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )


def permuted_sequence_adder(rowgen: RowGen,
                            num_sequences: int,
                            keep_pos: bool = False,
                            random_state: Optional[np.random.RandomState] = None
                           ) -> Generator[DataSetGAN, DataSetGAN, None]:
    """

    Args:
        rowgen: Used for **pre-populating** the generator only!
        num_sequences: Number of sequences to generate in each iteration.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    seq_buffer = [row_to_dataset(r, 0).seq for r in itertools.islice(rowgen, 512)]
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        seq = dsg.seqs[0]
        negative_seq_big = b''.join(seq_buffer)
        negative_seqs = []
        for _ in range(num_sequences):
            offset = random_state.randint(0, len(negative_seq_big) - len(seq))
            negative_seq = (negative_seq_big[offset:] + negative_seq_big[:offset])[:len(seq)]
            negative_seqs.append(negative_seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )
        # Reshuffle negative sequences
        seq_buffer.append(seq)
        random_state.shuffle(seq_buffer)
        random_state.pop()


def negative_sequence_adder(rowgen: RowGen,
                            method: str,
                            num_sequences: int,
                            keep_pos: bool = False,
                            random_state: Optional[np.random.RandomState] = None
                           ) -> Generator[DataSetGAN, DataSetGAN, None]:
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
            try:
                negative_ds = get_negative_example(ds, method, rowgen, random_state)
            except (exc.MaxNumberOfTriesExceededError, exc.SequenceTooLongError) as e:
                logger.error("Encountered error '%s' for dataset '%s'", e, ds)
                continue
            negative_seqs.append(negative_ds.seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )
