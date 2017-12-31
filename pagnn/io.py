import random
from pathlib import Path
from typing import Generator, List, NamedTuple, Optional

import numpy as np
import pyarrow.parquet as pq

from pagnn import get_adjacency


class SequenceTooShortError(Exception):
    pass


class _DataFrameRow(NamedTuple):
    qseq: str
    residue_idx_1_corrected: List[int]
    residue_idx_2_corrected: List[int]


class DataRow(NamedTuple):
    seqs: np.array
    adj: np.array
    targets: np.array


def iter_dataset_rows(parquet_root_folder: Path, domains, columns, weights: Optional[List[float]],  random_seed=None):
    # Get number of rows for each domain
    if weights is None:
        weights = [
            count_domain_rows(parquet_root_folder.joinpath(f'domain_id={domain}'))
            for domain in domains
        ]
    else:
        weights = weights[:]
    # Get generators that return one row at a time
    domain_generators = [
        iter_domain_rows(parquet_root_folder.joinpath(f'domain_id={domain}'), columns)
        for domain in domains
    ]
    # Yield one row at a time
    while domain_generators:
        cur_gen = random.choices(domain_generators, weights)[0]
        try:
            yield next(cur_gen)
        except StopIteration:
            idx = domain_generators.index(cur_gen)
            del weights[idx]
            del domain_generators[idx]


def iter_domain_rows(domain_folder: Path,
                     columns: Optional[List[str]] = None) -> Generator[_DataFrameRow, None, None]:
    """Iterate over parquet data in `domain_folder` row-by-row.

    Args:
        domain_folder: Location where domain-specific *.parquet files are stored.
        columns: Subset of the columns to load.
    """
    parquet_files = list(domain_folder.glob('*.parquet'))
    random.shuffle(parquet_files)
    for filepath in parquet_files:
        df = pq.read_table(filepath.as_posix(), columns=columns).to_pandas()
        df = df.reindex(np.random.permutation(df.index))
        for row in df.itertuples():
            yield row


def count_domain_rows(domain_folder: Path) -> int:
    parquet_files = list(domain_folder.glob('*.parquet'))
    num_rows = 0
    for filepath in parquet_files:
        df = pq.read_table(filepath.as_posix(), columns=['__index_level_0__']).to_pandas()
        num_rows += len(df)
    return num_rows


def row_to_dataset(row: _DataFrameRow, num_real: int = 1, num_fake: int = 1) -> DataRow:
    """Convert a dataset row to data that can be used as input for machine learning.

    Notes:
        * This function works under the assumption that there is one dataset per row.
    """
    seqs: List[str] = []
    # Real sequence
    seq = row.qseq.replace('-', '')
    if len(seq) < 16:
        raise SequenceTooShortError()
    seqs += [seq] * num_real
    # Fake sequence
    for _ in range(num_fake):
        offset = np.random.choice(np.arange(6, len(seq) - 6))
        seq2 = seq[offset:] + seq[:offset]
        seqs += [seq2]
    # Adjacency matrix
    adj = get_adjacency(row.qseq, row.residue_idx_1_corrected, row.residue_idx_2_corrected)
    # As arrays
    seqs = np.array(seqs, dtype=np.string_)
    targets = np.array([1] * num_real + [0] * num_fake)
    return DataRow(seqs, adj, targets)
