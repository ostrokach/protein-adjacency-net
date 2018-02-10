import functools
import logging
import pickle
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, Tuple

import numpy as np

from . import dataset, io, settings
from .exc import MaxNumberOfTriesExceededError, SequenceTooLongError
from .types import DataGen, DataRow, DataSet, DataSetCollection

logger = logging.getLogger(__name__)


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


def get_mutation_datagen(mutation_class: str, data_path: Path) -> DataGen:
    assert mutation_class in ['protherm', 'humsavar']

    if mutation_class == 'protherm':
        score_column = 'ddg_exp'
        parquet_file = (
            data_path.joinpath('protherm_dataset').joinpath('protherm_validaton_dataset.parquet'))
    elif mutation_class == 'humsavar':
        score_column = 'score_exp'
        parquet_file = (
            data_path.joinpath('mutation_datasets').joinpath('humsavar_validaton_dataset.parquet'))

    mutation_datarows = io.iter_domain_rows(
        parquet_file,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
            'qseq_mutation': 'mutation',
            score_column: 'score',
        })

    mutation_datasets = (dataset.row_to_dataset(row, target=1) for row in mutation_datarows)

    mutation_dsc = []
    for pos_ds in mutation_datasets:
        neg_seq = bytearray(pos_ds.seq)
        mutation = pos_ds.meta['mutation']
        mutation_idx = int(mutation[1:-1]) - 1
        assert neg_seq[mutation_idx] == ord(mutation[0]), (chr(neg_seq[mutation_idx]), mutation[0])
        neg_seq[mutation_idx] = ord(mutation[-1])
        neg_ds = DataSet(neg_seq, pos_ds.adj, pos_ds.meta['score'])
        mutation_dsc.append(([pos_ds], [neg_ds]))

    def datagen():
        for dvc in mutation_dsc:
            yield dvc

    return datagen


def permute_and_slice_datagen(datagen_pos: Iterator[DataRow],
                              datagen_neg: Optional[Generator[DataRow, Any, None]],
                              methods: Tuple) -> Iterator[DataSetCollection]:
    batch_pos = []
    assert 'permute' in methods
    slice_methods = [m for m in methods if m != 'permute']
    for i, row in enumerate(datagen_pos):
        dataset_pos = dataset.row_to_dataset(row, target=1)
        if len(dataset_pos.seq) < settings.MIN_SEQUENCE_LENGTH:
            continue
        batch_pos.append(dataset_pos)
        if (i + 1) % 256 == 0:
            batch_neg = dataset.add_permuted_examples(batch_pos)
            for pos, neg in zip(batch_pos, batch_neg):
                pos_list = [pos]
                neg_list = [neg]
                for method in slice_methods:
                    try:
                        other_neg = dataset.add_negative_example(
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
        dataset_pos = dataset.row_to_dataset(row, target=1)
        if len(dataset_pos.seq) < settings.MIN_SEQUENCE_LENGTH:
            continue
        datasets_neg = []
        try:
            for method in methods:
                dataset_neg = dataset.add_negative_example(
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

    datagen_pos = io.iter_dataset_rows(
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

    datagen_neg = io.iter_dataset_rows(
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
        with filepath.open('rb') as fin:
            d = pickle.load(fin)
            parquet_folder_weights = np.array([d[p.name] for p in parquet_folders])
        logger.info("Loaded folder weights from file: '%s'", filepath)
    except FileNotFoundError:
        logger.info("Generating folder weights for parquet folder: '%s'.", parquet_folders)
        parquet_folder_weights = io.get_weights(parquet_folders)
        d = {p.name: w for p, w in zip(parquet_folders, parquet_folder_weights)}
        with filepath.open('wb') as fout:
            pickle.dump(d, fout, pickle.HIGHEST_PROTOCOL)
    return parquet_folder_weights
