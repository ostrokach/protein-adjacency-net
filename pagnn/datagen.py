import functools
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple

import numpy as np

import pagnn
from pagnn import DataRow

from .dataset import DataSetCollection
from .exc import MaxNumberOfTriesExceededError, SequenceTooLongError

logger = logging.getLogger(__name__)

DataGen = Callable[[], Iterator[DataSetCollection]]
"""A function which returns an iterator over dataset collections."""


def get_datagen(subset: str,
                data_path: Path,
                min_seq_identity: int,
                methods: List[str],
                random_state: Optional[np.random.RandomState] = None) -> DataGen:
    """Return a function which can generate positive or negative training examples."""
    assert subset in ['training', 'validation', 'test']

    datagen_pos = _get_rowgen_pos(f'adjacency_matrix_{subset}_gt{min_seq_identity}.parquet',
                                  data_path, random_state)

    if len(methods) == 1 and 'permute' in methods:
        training_datagen = functools.partial(
            permute_and_slice_datagen, datagen_pos=datagen_pos, datagen_neg=None, methods=methods)
    else:
        datagen_neg = _get_rowgen_neg(
            f'adjacency_matrix_{subset}_gt{min_seq_identity}_gbseqlen.parquet', data_path,
            random_state)
        if 'permute' in methods:
            training_datagen = functools.partial(
                permute_and_slice_datagen,
                datagen_pos=datagen_pos,
                datagen_neg=datagen_neg,
                methods=methods)
        else:
            training_datagen = functools.partial(
                slice_datagen, datagen_pos=datagen_pos, datagen_neg=datagen_neg, methods=methods)

    return training_datagen


def permute_and_slice_datagen(datagen_pos: Iterator[DataRow],
                              datagen_neg: Optional[Generator[DataRow, Any, None]],
                              methods: Tuple) -> Iterator[DataSetCollection]:
    batch_pos = []
    assert 'permute' in methods
    slice_methods = [m for m in methods if m != 'permute']
    for i, row in enumerate(datagen_pos):
        dataset_pos = pagnn.row_to_dataset(row)
        if len(dataset_pos.seq) < pagnn.MIN_SEQUENCE_LENGTH:
            continue
        batch_pos.append(dataset_pos)
        if (i + 1) % 256 == 0:
            batch_neg = pagnn.add_permuted_examples(batch_pos)
            for pos, neg in zip(batch_pos, batch_neg):
                pos_list = [pos]
                neg_list = [neg]
                for method in slice_methods:
                    try:
                        other_neg = pagnn.add_negative_example(
                            pos,
                            method=method,
                            datagen=datagen_neg,
                        )
                        neg_list.append(other_neg)
                    except (MaxNumberOfTriesExceededError, SequenceTooLongError) as e:
                        logger.error("%s: %s", type(e), e)
                        continue
                yield pos_list, neg_list
            batch_pos = []


def slice_datagen(datagen_pos: Iterator[DataRow], datagen_neg: Generator[DataRow, Any, None],
                  methods: Tuple) -> Iterator[DataSetCollection]:
    for row in datagen_pos:
        dataset_pos = pagnn.row_to_dataset(row)
        if len(dataset_pos.seq) < pagnn.MIN_SEQUENCE_LENGTH:
            continue
        datasets_neg = []
        try:
            for method in methods:
                dataset_neg = pagnn.add_negative_example(
                    dataset_pos,
                    method=method,
                    datagen=datagen_neg,
                )
                datasets_neg.append(dataset_neg)
        except (MaxNumberOfTriesExceededError, SequenceTooLongError) as e:
            logger.error("%s: %s", type(e), e)
            continue
        yield [dataset_pos], datasets_neg


def _get_rowgen_pos(root_folder_name: str,
                    data_path: Path,
                    random_state: Optional[np.random.RandomState] = None) -> Iterator[DataRow]:
    parquet_folders = sorted(
        data_path.joinpath('threshold_by_pc_identity')
        .joinpath(root_folder_name).glob('database_id=*'))

    parquet_folder_weights_file = data_path.joinpath('threshold_by_pc_identity').joinpath(
        root_folder_name.partition('.')[0] + '-weights.pickle')
    parquet_folder_weights = _load_parquet_weights(parquet_folder_weights_file, parquet_folders)
    assert len(parquet_folders) == len(parquet_folder_weights)

    datagen_pos = pagnn.iter_dataset_rows(
        parquet_folders,
        parquet_folder_weights,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
        },
        random_state=random_state)

    return datagen_pos


def _get_rowgen_neg(root_folder_name: str,
                    data_path: Path,
                    random_state: Optional[np.random.RandomState] = None
                   ) -> Generator[DataRow, None, None]:
    parquet_folders = sorted(
        data_path.joinpath('group_by_sequence_length')
        .joinpath(root_folder_name).glob('qseq_length_bin=*'))

    parquet_folder_weights_file = data_path.joinpath('group_by_sequence_length').joinpath(
        root_folder_name.partition('.')[0] + '-weights.pickle')
    parquet_folder_weights = _load_parquet_weights(parquet_folder_weights_file, parquet_folders)
    assert len(parquet_folders) == len(parquet_folder_weights)

    datagen_neg = pagnn.iter_dataset_rows(
        parquet_folders,
        parquet_folder_weights,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
        },
        seq_length_constraint=True,
        random_state=random_state)
    next(datagen_neg)

    return datagen_neg


def _load_parquet_weights(filepath: Path, parquet_folders: List[Path]) -> np.ndarray:
    try:
        logger.info("Loading folder weights from file: '%s'", filepath)
        with filepath.open('rb') as fin:
            d = pickle.load(fin)
            parquet_folder_weights = np.array([d[p.name] for p in parquet_folders])
    except FileNotFoundError:
        logger.info("Loading folder weights from file failed! Recomputing...")
        parquet_folder_weights = pagnn.get_weights(parquet_folders)
        d = {p.name: w for p, w in zip(parquet_folders, parquet_folder_weights)}
        with filepath.open('wb') as fout:
            pickle.dump(d, fout, pickle.HIGHEST_PROTOCOL)
    return parquet_folder_weights
