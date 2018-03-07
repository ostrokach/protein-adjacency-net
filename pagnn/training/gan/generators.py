import argparse
import itertools
import logging
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
import tqdm

from pagnn import exc, settings
from pagnn.dataset import get_negative_example, row_to_dataset, to_gan
from pagnn.training.common import get_rowgen_mut, get_rowgen_neg, get_rowgen_pos
from pagnn.types import DataRow, DataSet, DataSetGAN

logger = logging.getLogger(__name__)

RowGen = Generator[DataRow, Tuple[Callable, int], None]

# === Dataset loaders ===


def get_validation_dataset(
        args: argparse.Namespace,
        method: str,
        data_path: Path,
        random_state: Optional[np.random.RandomState] = None) -> List[DataSetGAN]:

    if random_state is None:
        random_state = np.random.RandomState(42)

    positive_rowgen = get_rowgen_pos(
        'validation',
        args.validation_min_seq_identity,
        data_path,
        random_state=random_state,
    )

    negative_rowgen = get_rowgen_neg(
        'validation',
        args.validation_min_seq_identity,
        data_path,
        random_state=random_state,
    )

    nsa = negative_sequence_adder(
        negative_rowgen,
        method,
        num_sequences=1,
        keep_pos=True,
        random_state=random_state,
    )
    next(nsa)

    dataset = []
    with tqdm.tqdm(
            total=args.validation_num_sequences, desc=method,
            disable=not settings.SHOW_PROGRESSBAR) as progressbar:
        while len(dataset) < args.validation_num_sequences:
            pos_row = next(positive_rowgen)
            pos_ds = to_gan(row_to_dataset(pos_row, 1))
            ds = nsa.send(pos_ds)
            dataset.append(ds)
            progressbar.update(1)

    assert len(dataset) == args.validation_num_sequences
    return dataset


def get_mutation_dataset(mutation_class: str, data_path: Path) -> List[DataSetGAN]:

    mutation_datarows = get_rowgen_mut(mutation_class, data_path)
    mutation_datasets = (to_gan(row_to_dataset(row, target=1)) for row in mutation_datarows)

    mutation_dsg = []
    for pos_ds in mutation_datasets:
        neg_seq = bytearray(pos_ds.seqs[0])
        mutation = pos_ds.meta['mutation']
        mutation_idx = int(mutation[1:-1]) - 1
        assert neg_seq[mutation_idx] == ord(mutation[0]), (chr(neg_seq[mutation_idx]), mutation[0])
        neg_seq[mutation_idx] = ord(mutation[-1])
        ds = pos_ds._replace(
            seqs=pos_ds.seqs + [neg_seq],
            targets=pos_ds.targets + [pos_ds.meta['score']],
        )
        mutation_dsg.append(ds)

    return mutation_dsg


# === Negative dataset generators ===


def chain_generators(gens: List[Generator[DataSetGAN, DataSetGAN, None]]
                    ) -> Generator[DataSetGAN, DataSetGAN, None]:
    ds = None
    while True:
        ds = yield ds
        for gen in gens:
            ds = gen.send(ds)


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
    MAX_TRIES = 5
    if random_state is None:
        random_state = np.random.RandomState()
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        ds = DataSet(dsg.seqs[0], dsg.adjs[0], dsg.targets[0])
        negative_seqs = []
        n_tries = 0
        while len(negative_seqs) < num_sequences:
            try:
                negative_ds = get_negative_example(ds, method, rowgen, random_state)
            except (exc.MaxNumberOfTriesExceededError, exc.SequenceTooLongError) as e:
                logger.error("Encountered error '%s' for dataset '%s'", e, ds)
                if n_tries < MAX_TRIES:
                    n_tries += 1
                    continue
                else:
                    raise
            negative_seqs.append(negative_ds.seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )
