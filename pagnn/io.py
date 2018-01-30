import logging
from pathlib import Path
from typing import Generator, List, NamedTuple, Optional, Union

import numpy as np
import pyarrow.parquet as pq

from kmtools import py_tools

logger = logging.getLogger(__name__)


class DataRow(NamedTuple):
    sequence: str
    adjacency_idx_1: List[int]
    adjacency_idx_2: List[int]


def iter_dataset_rows(parquet_folders: List[Path],
                      weights: Optional[np.ndarray] = None,
                      columns: Union[dict, tuple] = DataRow._fields,
                      filters: tuple = (),
                      random_state: Optional[np.random.RandomState] = None,
                      seq_length_constraint=None) -> Generator[DataRow, None, None]:
    """Iterate over data from Parquet files in multiple `parquet_folders`.

    Notes:
        - Generating a training dataset from multiple domain folders
          is better than preshuffling and storing all domains in a single
          Parquet folder because it's easier to make each epoch trully random.

    Args:
        parquet_folders: List of domain folders from which to obtain data.
        weights: Number of rows in all Parquet files in each folder in `parquet_folders`.

    Returns:
        NamedTuple containing domain rows.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    weights = weights or _get_weights(parquet_folders)
    assert weights.sum() == 1

    generators = np.array([
        py_tools.iter_forever(lambda: iter_domain_rows(parquet_folder, columns, filters))
        for parquet_folder in parquet_folders
    ])

    if seq_length_constraint is None:
        yield from _iter_dataset_rows(generators, weights, random_state)
    else:
        seq_lengths = np.array(
            [_get_min_sequence_size(parquet_folder) for parquet_folder in parquet_folders])
        yield from _iter_dataset_rows_with_constraint(generators, weights, seq_lengths,
                                                      random_state)


def iter_domain_rows(
        parquet_folder: Path,
        columns: Union[dict, tuple] = DataRow._fields,
        filters: tuple = (),
        random_state: Optional[np.random.RandomState] = None) -> Generator[DataRow, None, None]:
    """Iterate over parquet data in `parquet_folder` row-by-row.

    Args:
        parquet_folder: Location where domain-specific *.parquet files are stored.
        columns: Additional columns that should be present in the returned `row` NamedTuples.
        filters: Tuple of functions with take a single `row` as input and return `True`
                 if that row should be kept.

    Yields:
        NamedTuple containing domain rows.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    # Validate parameters
    if isinstance(columns, dict):
        column_renames = {c: c2 for c, c2 in columns.items() if c2 is not None}
    else:
        column_renames = {}
    # Get a list of Parquet files from domain folder
    parquet_files = _get_domain_parquet_files(parquet_folder)
    assert parquet_files, parquet_folder
    # Iterate over parquet files
    random_state.shuffle(parquet_files)
    for filepath in parquet_files:
        parquet_file = pq.ParquetFile(filepath.as_posix())
        for row_group_idx in range(parquet_file.num_row_groups):
            df = parquet_file.read_row_group(row_group_idx, columns=list(columns)).to_pandas()
            df = df.reindex(random_state.permutation(df.index)).rename(columns=column_renames)
            assert not set(DataRow._fields) - set(df.columns)
            for row in df.itertuples():
                if not all(f(row) for f in filters):
                    continue
                yield row


def count_domain_rows(parquet_folder: Path) -> int:
    parquet_files = _get_domain_parquet_files(parquet_folder)
    num_rows = 0
    for filepath in parquet_files:
        df = pq.read_table(filepath.as_posix(), columns=['__index_level_0__']).to_pandas()
        num_rows += len(df)
    return num_rows


def _get_weights(parquet_folders: List[Path]) -> np.ndarray:
    logger.debug("Generating weights for domain folders...")
    if len(parquet_folders) > 1:
        weights = np.array(
            [count_domain_rows(parquet_folder) for parquet_folder in parquet_folders])
    else:
        weights = np.array([1])
    logger.debug("Done generating weights!")
    weights = weights[:] / weights.sum()


def _iter_dataset_rows(generators, weights, random_state):
    while True:
        yield next(random_state.choice(generators, replace=False, p=weights))


def _iter_dataset_rows_with_constraint(generators, weights, constraint_array, random_state):
    row = None
    while True:
        op, target_seq_length = (yield row)
        idx = op(constraint_array, target_seq_length)
        cur_generators = generators[idx]
        cur_weights = weights[idx] / weights[idx].sum()
        cur_gen = random_state.choice(cur_generators, replace=False, p=cur_weights)
        row = next(cur_gen)


def _get_min_sequence_size(parquet_folder: Path) -> int:
    return int(parquet_folder.name[16:])


def _get_domain_parquet_files(parquet_folder: Path) -> List[Path]:
    if parquet_folder.is_file():
        parquet_files = [parquet_folder]
    else:
        parquet_files = list(parquet_folder.glob('**/*.parquet'))
    return parquet_files
