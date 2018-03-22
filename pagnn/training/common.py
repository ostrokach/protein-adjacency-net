import logging
import pickle
from pathlib import Path
from typing import Generator, Iterator, List, Optional

import numpy as np

from pagnn.io import get_folder_weights, iter_datarows, iter_datarows_shuffled
from pagnn.types import DataRow

logger = logging.getLogger(__name__)


def get_rowgen_pos(subset: str,
                   min_seq_identity: int,
                   data_path: Path,
                   random_state: Optional[np.random.RandomState] = None) -> Iterator[DataRow]:
    """Return an iterator of `DataRow` objects from the positive training dataset.

    Thin wrapper around `pagnn.io.iter_datarows_shuffled` providing default parameters.
    """
    assert subset in ['training', 'validation', 'test']
    assert min_seq_identity in [0, 40, 60, 80]
    root_folder_name = f'adjacency_matrix_{subset}_gt{min_seq_identity}.parquet'

    parquet_folders = sorted(
        data_path.joinpath('threshold_by_pc_identity')
        .joinpath(root_folder_name).glob('database_id=*'))

    parquet_folder_weights_file = data_path.joinpath('threshold_by_pc_identity').joinpath(
        root_folder_name.partition('.')[0] + '-weights.pickle')
    parquet_folder_weights = load_parquet_weights(parquet_folder_weights_file, parquet_folders)
    assert len(parquet_folders) == len(parquet_folder_weights)

    rowgen_pos = iter_datarows_shuffled(
        parquet_folders,
        parquet_folder_weights,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
        },
        random_state=random_state)

    return rowgen_pos


def get_rowgen_neg(
        subset: str,
        min_seq_identity: int,
        data_path: Path,
        random_state: Optional[np.random.RandomState] = None) -> Generator[DataRow, None, None]:
    """Return an iterator of `DataRow` objects from the negative training dataset.

    Thin wrapper around `pagnn.io.iter_datarows_shuffled` providing default parameters.
    """
    assert subset in ['training', 'validation', 'test']
    assert min_seq_identity in [0, 40, 60, 80]
    root_folder_name = f'adjacency_matrix_{subset}_gt{min_seq_identity}_gbseqlen.parquet'

    parquet_folders = sorted(
        data_path.joinpath('group_by_sequence_length')
        .joinpath(root_folder_name).glob('qseq_length_bin=*'))

    parquet_folder_weights_file = data_path.joinpath('group_by_sequence_length').joinpath(
        root_folder_name.partition('.')[0] + '-weights.pickle')
    parquet_folder_weights = load_parquet_weights(parquet_folder_weights_file, parquet_folders)
    assert len(parquet_folders) == len(parquet_folder_weights)

    rowgen_neg = iter_datarows_shuffled(
        parquet_folders,
        parquet_folder_weights,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
        },
        seq_length_constraint=True,
        random_state=random_state)
    next(rowgen_neg)

    return rowgen_neg


def get_rowgen_mut(
        mutation_class: str,
        data_path: Path,
):
    assert mutation_class in ['protherm', 'humsavar']

    if mutation_class == 'protherm':
        score_column = 'ddg_exp'
        parquet_file = (
            data_path.joinpath('protherm_dataset').joinpath('protherm_validaton_dataset.parquet'))
    elif mutation_class == 'humsavar':
        score_column = 'score_exp'
        parquet_file = (
            data_path.joinpath('mutation_datasets').joinpath('humsavar_validaton_dataset.parquet'))

    rowgen_mut = iter_datarows(
        parquet_file,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
            'qseq_mutation': 'mutation',
            score_column: 'score',
        })

    return rowgen_mut


def load_parquet_weights(filepath: Path, parquet_folders: List[Path]) -> np.ndarray:
    """Load folder weights from a pickle file in a predefined location."""
    try:
        with filepath.open('rb') as fin:
            d = pickle.load(fin)
            parquet_folder_weights = np.array([d[p.name] for p in parquet_folders])
        logger.info("Loaded folder weights from file: '%s'", filepath)
    except FileNotFoundError:
        logger.info("Generating folder weights for parquet folders: '%s'.", parquet_folders)
        parquet_folder_weights = get_folder_weights(parquet_folders)
        d = {p.name: w for p, w in zip(parquet_folders, parquet_folder_weights)}
        with filepath.open('wb') as fout:
            pickle.dump(d, fout, pickle.HIGHEST_PROTOCOL)
    return parquet_folder_weights
