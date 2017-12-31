import random
from pathlib import Path
from typing import Generator, List, NamedTuple, Optional

import numpy as np
import pyarrow.parquet as pq

from pagnn import permute_sequence, get_adjacency


class SequenceTooShortError(Exception):
    pass


class _DataFrameRow(NamedTuple):
    qseq: str
    residue_idx_1_corrected: List[int]
    residue_idx_2_corrected: List[int]


class DataRow(NamedTuple):
    seq: np.ndarray
    adj: np.ndarray
    targets: np.ndarray
    seq_lengths: np.ndarray


def iter_datasets(domain_folders: List[Path],
                  max_seq_len: int=100_000) -> Generator[DataRow, None, None]:
    """Generate datasets by randomly combining sequences from different domains.

    Args:
        domain_folders: List of domain folders containing parquet files
            with required columns.
        max_seq_len: Maximum number of residues that a sequence example can have.

    Yields:
        Datasets suitable for machine learning.
    """
    batch: List[_DataFrameRow] = []
    batch_len = 0
    for row in iter_dataset_rows(domain_folders):
        row_len = len(row.qseq.replace('-', ''))
        if batch_len + row_len <= max_seq_len:
            batch.append(row)
            batch_len += row_len
        else:
            dataset = rows_to_dataset(batch)
            yield dataset
            batch = [row]
            batch_len = row_len


def rows_to_dataset(rows: List[_DataFrameRow]) -> DataRow:
    """Combine one or more rows into a dataset."""
    seq_list: List[str] = []
    adj_list: List[np.ndarray] = []
    # Real sequence and adjacency matrix
    for row in rows:
        seq = row.qseq.replace('-', '')
        seq_list.append(seq)
        adj = get_adjacency(row.qseq, row.residue_idx_1_corrected, row.residue_idx_2_corrected)
        adj_list.append(adj)
    # Real and fake sequences
    seq_pos = ''.join(seq_list)
    seq_neg = permute_sequence(seq_pos)
    # Combine data from all batches
    seq_array = np.array([[seq_pos], [seq_neg]], dtype=np.string_)
    adj = np.hstack(adj_list)
    targets = np.array([1, 0] * len(seq_list))
    seq_lengths = np.array([len(seq) for seq in seq_list], dtype=np.int8)
    return DataRow(seq_array, adj, targets, seq_lengths)


def iter_dataset_rows(domain_folders,
                      weights: Optional[List[float]]=None) -> Generator[_DataFrameRow, None, None]:
    # Get number of rows for each domain
    if weights is None:
        weights = [count_domain_rows(domain_folder) for domain_folder in domain_folders]
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
        columns: Subset of the columns to load.
    """
    parquet_files = list(domain_folder.glob('*.parquet'))
    random.shuffle(parquet_files)
    for filepath in parquet_files:
        df = pq.read_table(
            filepath.as_posix(),
            columns=['qseq', 'residue_idx_1_corrected', 'residue_idx_2_corrected']).to_pandas()
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
