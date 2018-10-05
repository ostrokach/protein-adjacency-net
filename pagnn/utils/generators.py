import itertools
import logging
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import torch

from pagnn import exc
from pagnn.dataset import get_negative_example, get_offset, row_to_dataset
from pagnn.io import iter_datarows
from pagnn.types import DataSet, DataSetGAN, RowGen, RowGenF

logger = logging.getLogger(__name__)


def get_rowgen_mut(mutation_class: str, data_path: Path):
    assert mutation_class in ["protherm", "humsavar"]

    if mutation_class == "protherm":
        score_column = "ddg_exp"
        parquet_file = data_path.joinpath("protherm_dataset").joinpath(
            "protherm_validaton_dataset.parquet"
        )
    elif mutation_class == "humsavar":
        score_column = "score_exp"
        parquet_file = data_path.joinpath("mutation_datasets").joinpath(
            "humsavar_validaton_dataset.parquet"
        )

    rowgen_mut = iter_datarows(
        parquet_file,
        columns={
            "qseq": "sequence",
            "residue_idx_1_corrected": "adjacency_idx_1",
            "residue_idx_2_corrected": "adjacency_idx_2",
            "qseq_mutation": "mutation",
            score_column: "score",
        },
    )

    return rowgen_mut


def basic_permuted_sequence_adder(
    num_sequences: int, keep_pos: bool, random_state: Optional[np.random.RandomState] = None
):
    if random_state is None:
        random_state = np.random.RandomState()

    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        seq = dsg.seqs[0]
        negative_seqs = []
        for _ in range(num_sequences):
            offset = get_offset(seq.shape[1], random_state)
            negative_seq = torch.sparse_coo_tensor(
                torch.cat([seq._indices()[:, offset:], seq._indices()[:, :offset]], 1),
                seq._values(),
                size=seq.size(),
            )
            assert (
                seq.size() == negative_seq.size()
                and seq._values().size() == negative_seq._values().size()
                and seq._indices().size() == negative_seq._indices().size()
            )
            negative_seqs.append(negative_seq)
        negative_targets = torch.zeros(num_sequences, dtype=torch.float)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            # adjs=(dsg.adjs if keep_pos else []) + dsg.adjs * len(negative_seqs),
            targets=torch.cat([dsg.targets, negative_targets]) if keep_pos else negative_targets,
        )


def buffered_permuted_sequence_adder(
    rowgen: RowGen,
    num_sequences: int,
    keep_pos: bool = False,
    random_state: Optional[np.random.RandomState] = None,
) -> Generator[Optional[DataSetGAN], DataSetGAN, None]:
    """

    Args:
        rowgen: Used for **pre-populating** the generator only!
        num_sequences: Number of sequences to generate in each iteration.
    """
    raise NotImplementedError

    if random_state is None:
        random_state = np.random.RandomState()

    seq_buffer = [row_to_dataset(r, 0).seq for r in itertools.islice(rowgen, 512)]
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        seq = dsg.seqs[0]
        negative_seq_big = b"".join(seq_buffer)
        negative_seqs = []
        for _ in range(num_sequences):
            offset = random_state.randint(0, len(negative_seq_big) - len(seq))
            negative_seq = (negative_seq_big[offset:] + negative_seq_big[:offset])[: len(seq)]
            negative_seqs.append(negative_seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )
        # Reshuffle negative sequences
        seq_buffer.append(seq)
        random_state.shuffle(seq_buffer)
        random_state.pop()


def negative_sequence_adder(
    rowgen: RowGenF,
    method: str,
    num_sequences: int,
    keep_pos: bool = False,
    random_state: Optional[np.random.RandomState] = None,
) -> Generator[Optional[DataSetGAN], DataSetGAN, None]:
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
        assert ds.seq.shape[1] == ds.seq._indices().shape[1] == ds.seq._values().shape[0]
        negative_seqs: List[bytes] = []
        succeeded = True
        while len(negative_seqs) < num_sequences:
            try:
                negative_ds = get_negative_example(ds, method, rowgen, random_state)
            except (exc.MaxNumberOfTriesExceededError, exc.SequenceTooLongError) as e:
                logger.error("Encountered error '%s' for dataset '%s'", e, ds)
                succeeded = False
                break
            negative_seqs.append(negative_ds.seq)
        if succeeded:
            negative_dsg = dsg._replace(
                seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
                targets=(
                    torch.cat([dsg.targets, torch.zeros(num_sequences, dtype=torch.float)])
                    if keep_pos
                    else torch.zeros(num_sequences, dtype=torch.float)
                ),
            )
        else:
            negative_dsg = None
