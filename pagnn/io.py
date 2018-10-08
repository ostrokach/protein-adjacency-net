"""Functions for loading training and validation data.

Training and validation data are stored in *Parquet* files.
"""
import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pagnn.types import DataRow, RowGen, RowGenF

logger = logging.getLogger(__name__)


# === Functions for reading single Parquet files ===


def iter_datarows(
    parquet_file: Path,
    columns: Union[dict, tuple] = DataRow._fields,
    filters: List[Callable] = [],
    random_state: Optional[np.random.RandomState] = None,
) -> RowGen:
    """Iterate over rows in `parquet_file` in pseudo-random order.

    Args:
        file: Location where the domain-specific ``*.parquet`` file is stored.
        columns: Additional columns that should be present in the returned `row` NamedTuples.
        filters:

    Yields:
        NamedTuple containing domain rows.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    while True:
        df = _read_random_row_group(parquet_file, columns, filters, random_state)
        for row in df.itertuples():
            yield row


def gen_datarows(
    parquet_file: Path,
    columns: Union[dict, tuple] = DataRow._fields,
    filters: List[Callable] = [],
    random_state: Optional[np.random.RandomState] = None,
) -> RowGenF:
    """Iterate over rows in `parquet_file` in pseudo-random order.

    Args:
        file: Location where the domain-specific ``*.parquet`` file is stored.
        columns: Additional columns that should be present in the returned `row` NamedTuples.
        filters:

    Yields:
        NamedTuple containing domain rows.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    fn = yield None
    while True:
        df = _read_random_row_group(parquet_file, columns, filters, random_state)
        if fn is not None:
            df = fn(df)
        tup = next(df.itertuples()) if not df.empty else None
        fn = yield tup


def _read_random_row_group(
    parquet_file: Path,
    columns: Union[dict, tuple],
    filters: List[Callable],
    random_state: np.random.RandomState,
) -> pd.DataFrame:
    """Read a random row group from an open `parquet_file_obj`.

    TODO: Refactor this ugly function to take fewer arguments.
    """
    column_renames = _get_column_renames(columns)
    parquet_file_obj = pq.ParquetFile(parquet_file)
    row_group_idx = random_state.randint(parquet_file_obj.num_row_groups)
    logger.debug("Reading row group %s from parquet file '%s'.", row_group_idx, parquet_file)
    table = parquet_file_obj.read_row_group(
        row_group_idx, columns=list(columns), nthreads=len(columns)
    )
    df = table.to_pandas(use_threads=True)
    df = df.rename(columns=column_renames)
    for fn in filters:
        df = fn(df)
    df = df.reindex(random_state.permutation(df.index))
    assert not set(DataRow._fields) - set(df.columns)
    return df


def _get_column_renames(columns: Union[dict, tuple]) -> dict:
    if isinstance(columns, dict):
        column_renames = {c: c2 for c, c2 in columns.items() if c2 is not None}
    else:
        column_renames = {}
    return column_renames


# === Functions for reading collections of Parquet files ===


def iter_datarows_shuffled(
    parquet_files: List[Path],
    columns: Union[dict, tuple] = DataRow._fields,
    filters: List[Callable] = [],
    random_state: Optional[np.random.RandomState] = None,
) -> RowGen:
    """Iterate over parquet data in multiple `parquet_files`, randomly chosing the next row.

    Notes:
        - Generating a training dataset from multiple domain folders
          is better than preshuffling and storing all domains in a single
          Parquet folder because it's easier to make each epoch trully random.

    Args:
        parquet_files: List of parquet files from which to obtain data.

    Returns:
        NamedTuple containing domain rows.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    # Note: right now we are not considering the effect of `filters`
    # weights = np.array(
    #     [pq.ParquetFile(f).scan_contents(columns=["__index_level_0__"]) for f in parquet_files]
    # )

    weights_list = []
    generator_list = []
    for parquet_file in parquet_files:
        weight = pq.ParquetFile(parquet_file).metadata.num_rows
        generator = iter_datarows(parquet_file, columns, filters, random_state)
        weights_list.append(weight)
        generator_list.append(generator)

    weights = np.array(weights_list)
    weights = weights[:] / weights.sum()

    generators = np.array(generator_list)

    while True:
        gen = random_state.choice(generators, p=weights)
        yield next(gen)


def gen_datarows_shuffled(
    parquet_files: List[Path],
    columns: Union[dict, tuple] = DataRow._fields,
    filters: List[Callable] = [],
    random_state: Optional[np.random.RandomState] = None,
) -> RowGenF:
    """."""
    if random_state is None:
        random_state = np.random.RandomState()

    weights_list = []
    generator_list = []
    for parquet_file in parquet_files:
        weight = pq.ParquetFile(parquet_file).metadata.num_rows
        generator = gen_datarows(parquet_file, columns, filters, random_state)
        next(generator)
        weights_list.append(weight)
        generator_list.append(generator)

    weights = np.array(weights_list)
    weights = weights[:] / weights.sum()

    generators = np.array(generator_list)

    input_ = yield None
    while True:
        gen = random_state.choice(generators, p=weights)
        input_ = yield gen.send(input_)
