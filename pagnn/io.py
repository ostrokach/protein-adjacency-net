import logging
import random
from pathlib import Path
from typing import Generator, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from pagnn import GAP_LENGTH, get_adjacency

logger = logging.getLogger(__name__)


class SequenceTooShortError(Exception):
    pass


class _DataFrameRow(NamedTuple):
    qseq: str
    residue_idx_1_corrected: List[int]
    residue_idx_2_corrected: List[int]


class DataRow(NamedTuple):
    seq: np.ndarray
    adjs: List[np.ndarray]
    targets: np.ndarray


def iter_datasets(rows: Iterable[_DataFrameRow], max_seq_len: int = 10_000
                 ) -> Generator[Tuple[bytes, List[np.ndarray]], None, None]:
    """Combine several rows into a single dataset.

    Args:
        rows: List or iterator of rows.
        max_seq_len: Maximum number of residues that a sequence example can have.

    Yields:
        Inputs for machine learning.
    """
    batch: List[_DataFrameRow] = []
    batch_len = 0
    for row in rows:
        if batch_len + len(row.qseq) <= max_seq_len:
            batch.append(row)
            batch_len += len(row.qseq)
        else:
            yield gen_dataset(batch)
            batch = [row]
            batch_len = len(row.qseq)
    yield gen_dataset(batch)


def gen_dataset(rows: Iterable[_DataFrameRow]) -> Tuple[bytes, List[np.ndarray]]:
    """Combine one or more rows into a dataset."""
    seq = (b'X' * GAP_LENGTH).join(row.qseq.replace('-', '').encode('ascii') for row in rows)
    adjs = [
        get_adjacency(row.qseq, row.residue_idx_1_corrected, row.residue_idx_2_corrected)
        for row in rows
    ]
    return seq, adjs


def iter_dataset_rows(domain_folders, weights: Optional[List[float]] = None
                     ) -> Generator[_DataFrameRow, None, None]:
    # Get number of rows for each domain
    if weights is None:
        logger.debug("Generating weights for domain folders...")
        if len(domain_folders) > 1:
            weights = [count_domain_rows(domain_folder) for domain_folder in domain_folders]
        else:
            weights = [1]
        logger.debug("Done generating weights!")
    else:
        weights = weights[:]
    # Get generators that return one row at a time
    domain_generators = [iter_domain_rows(domain_folder) for domain_folder in domain_folders]
    # Yield one row at a time
    while domain_generators:
        cur_gen = random.choices(domain_generators, weights)[0]
        try:
            yield next(cur_gen)
        except StopIteration:
            # Remove exhausted generator from options
            idx = domain_generators.index(cur_gen)
            del weights[idx]
            del domain_generators[idx]


def iter_domain_rows(domain_folder: Path) -> Generator[_DataFrameRow, None, None]:
    """Iterate over parquet data in `domain_folder` row-by-row.

    Args:
        domain_folder: Location where domain-specific *.parquet files are stored.

    Yields:
        NamedTuple containing domain rows.
    """
    parquet_files = _get_domain_parquet_files(domain_folder)
    assert parquet_files
    random.shuffle(parquet_files)
    for filepath in parquet_files:
        df = pq.read_table(
            filepath.as_posix(),
            columns=['qseq', 'residue_idx_1_corrected', 'residue_idx_2_corrected']).to_pandas()
        df = df.reindex(np.random.permutation(df.index))
        for row in df.itertuples():
            yield row


def count_domain_rows(domain_folder: Path) -> int:
    parquet_files = _get_domain_parquet_files(domain_folder)
    num_rows = 0
    for filepath in parquet_files:
        df = pq.read_table(filepath.as_posix(), columns=['__index_level_0__']).to_pandas()
        num_rows += len(df)
    return num_rows


def _get_domain_parquet_files(domain_folder: Path) -> List[Path]:
    if domain_folder.is_file():
        parquet_files = [domain_folder]
    else:
        parquet_files = list(domain_folder.glob('**/*.parquet'))
    return parquet_files
