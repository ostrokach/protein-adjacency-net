import random
from pathlib import Path
from typing import List, NamedTuple, Generator

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


def iter_domain_rows(domain_folder: Path, columns=None,
                     subsample=None) -> Generator[_DataFrameRow, None, None]:
    parquet_files = list(domain_folder.glob('*.parquet'))
    row_idx = 0
    for filepath in parquet_files:
        df = pq.read_table(filepath.as_posix(), columns=columns).to_pandas()
        if subsample:
            num_samples = min(len(df), subsample // len(parquet_files) + 1)
            subsample_idxs = random.sample(range(len(df)), k=num_samples)
            df = df.iloc[subsample_idxs]
        for row in df.itertuples():
            yield row
            row_idx += 1
            if subsample and row_idx > subsample:
                return


def row_to_dataset(row: _DataFrameRow, num_real: int = 1, num_fake: int = 1) -> DataRow:
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
