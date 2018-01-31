import operator
from typing import Callable, Generator, Iterable, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from pagnn import DataRow, get_adj_identity, get_adjacency, get_seq_identity

MAX_TRIES = 64
MAX_TRIES_SEQLEN = 1024


class MaxNumberOfTriesExceededError(Exception):
    pass


class DataSet(NamedTuple):
    seq: bytes
    adj: sparse.spmatrix
    target: float


# === Positive training examples ===


def row_to_dataset(row: DataRow) -> DataSet:
    seq = row.sequence.replace('-', '').encode('ascii')
    return DataSet(seq, get_adjacency(len(seq), row.adjacency_idx_1, row.adjacency_idx_2), 1)


def iter_datasets(rows: Iterable[DataRow]) -> Iterator[DataSet]:
    """Convert an iterable over rows into an iterable over datasets."""
    for row in rows:
        yield row_to_dataset(row)


def iter_dataset_batches(rows: Iterable[DataRow],
                         max_seq_len: int = 50_000) -> Iterator[List[DataSet]]:
    """Combine several rows into a single dataset.

    Args:
        rows: List or iterator of rows.
        max_seq_len: Maximum number of residues that a sequence example can have.

    Yields:
        Inputs for machine learning.
    """
    batch: List[DataSet] = []
    batch_len = 0
    for row in rows:
        dataset = row_to_dataset(row)
        if batch_len + len(dataset.seq) <= max_seq_len:
            batch.append(dataset)
            batch_len += len(dataset.seq)
        else:
            yield batch
            batch = [dataset]
            batch_len = len(dataset.seq)
    yield batch


# === Negative training examples ===


def add_negative_example(datasets: List[DataSet],
                         method='permute',
                         datagen: Optional[Generator[DataRow, Tuple[Callable, int], None]] = None,
                         random_state: Optional[np.random.RandomState] = None) -> List[DataSet]:
    assert method in ['permute', 'start', 'stop', 'middle', 'edges', 'exact']

    negative_datasets: List[DataSet] = []
    if method == 'permute':
        combined_seq = b''.join(ds.seq for ds in datasets)
        combined_adj = sparse.block_diag([ds.adj for ds in datasets])
        # Permute a sequence until we find something with low sequence identity
        n_tries = 0
        seq_identity = 1.0
        adj_identity = 1.0
        while seq_identity > 0.2 or adj_identity > 0.2:
            n_tries += 1
            if n_tries > MAX_TRIES:
                raise MaxNumberOfTriesExceededError()
            offset = get_offset(len(combined_seq))
            # Permute sequence
            negative_seq = combined_seq[offset:] + combined_seq[:offset]
            seq_identity = get_seq_identity(combined_seq, negative_seq)
            # Permute adjacencies
            negative_adj = _permute_adjacency(combined_adj, offset)
            adj_identity = get_adj_identity(combined_adj, negative_adj)
        negative_seqs = _split_sequence(negative_seq, [len(ds.seq) for ds in datasets])
        negative_adjs = _split_adjacency(negative_adj, [ds.adj.shape[-1] for ds in datasets])
        for nseq, nadj in zip(negative_seqs, negative_adjs):
            negative_datasets.append(DataSet(nseq, nadj, 0))
    else:
        for ds in datasets:
            # Find a negative sequence with low sequence identity
            n_tries = 0
            seq_identity = 1.0
            adj_identity = 1.0
            while seq_identity > 0.2 or adj_identity > 0.2:
                n_tries += 1
                if n_tries > MAX_TRIES:
                    raise MaxNumberOfTriesExceededError()
                if method == 'exact':
                    n_tries_seqlen = 0
                    while True:
                        n_tries_seqlen += 1
                        if n_tries_seqlen > MAX_TRIES_SEQLEN:
                            raise MaxNumberOfTriesExceededError()
                        row = datagen.send((operator.eq, int(len(ds.seq) // 20 * 20)))
                        if len(row.sequence.replace('-', '')) == len(ds.seq):
                            break
                else:
                    row = datagen.send((operator.ge, len(ds.seq) + 20))
                template_seq = row.sequence.replace('-', '')
                start, stop = get_indices(len(ds.seq), len(template_seq), method, random_state)
                if method in ['exact', 'start', 'stop', 'middle']:
                    negative_seq = (template_seq[start:stop]).encode('ascii')
                    cut_adjacency_idx_1, cut_adjacency_idx_2 = _extract_adjacency_from_middle(
                        start, stop, row.adjacency_idx_1, row.adjacency_idx_2)
                else:
                    negative_seq = (template_seq[:start] + template_seq[stop:]).encode('ascii')
                    cut_adjacency_idx_1, cut_adjacency_idx_2 = _extract_adjacency_from_edges(
                        start, stop, row.adjacency_idx_1, row.adjacency_idx_2)
                seq_identity = get_seq_identity(ds.seq, negative_seq)
                negative_adj = get_adjacency(ds.adj.shape[0], cut_adjacency_idx_1,
                                             cut_adjacency_idx_2)
                adj_identity = get_adj_identity(ds.adj, negative_adj)
            negative_datasets.append(DataSet(negative_seq, negative_adj, 0))

    assert len(negative_datasets) == len(datasets)

    return negative_datasets


def interpolate_sequences(
        positive_seq: bytes,
        negative_seq: bytes,
        interpolate: int = 0,
        random_state: Optional[np.random.RandomState] = None) -> Tuple[List[bytes], List[float]]:
    """
    Examples:
        >>> interpolated_seqs, interpolated_targets = interpolate_sequences(
        ...     b'AAAAA', b'BBBBB', 4,  np.random.RandomState(42))
        >>> [round(f, 3) for f in interpolated_targets]
        [0.8, 0.6, 0.4, 0.2]
        >>> interpolated_seqs
        [b'AAABA', b'BAAAB', b'AABBB', b'ABBBB']
    """
    if random_state is None:
        random_state = np.random.RandomState()
    interpolated_seqs = []
    fraction_positive = np.linspace(1, 0, interpolate + 2)[1:-1]
    if interpolate:
        for i in range(interpolate):
            fp = fraction_positive[i]
            idx = random_state.choice(
                np.arange(len(positive_seq)), int(round(fp * len(positive_seq))), replace=False)
            seq = bytearray(negative_seq)
            for i in idx:
                seq[i] = positive_seq[i]
            interpolated_seqs.append(bytes(seq))
    return interpolated_seqs, fraction_positive.tolist()


def interpolate_adjacencies(positive_adj: sparse.spmatrix,
                            negative_adj: sparse.spmatrix,
                            interpolate: int = 0,
                            random_state: Optional[np.random.RandomState] = None
                           ) -> Tuple[List[sparse.spmatrix], List[float]]:
    """

    Examples:
        >>> from scipy import sparse
        >>> positive_adj = sparse.coo_matrix((
        ...     np.ones(11), (
        ...         np.r_[np.arange(7), [0, 0, 5, 6]],
        ...         np.r_[np.arange(7), [5, 6, 0, 0]]), ))
        >>> negative_adj = sparse.coo_matrix((np.ones(7), (np.arange(7), np.arange(7)), ))
        >>> interpolated_adjs, interpolated_targets = interpolate_adjacencies(
        ...     positive_adj, negative_adj, 1, np.random.RandomState(42))
        >>> interpolated_targets
        [0.5]
        >>> interpolated_adjs[0].row
        array([0, 0, 1, 2, 3, 4, 5, 6, 6], dtype=int32)
        >>> interpolated_adjs[0].col
        array([0, 6, 1, 2, 3, 4, 5, 0, 6], dtype=int32)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    interpolated_adjs = []
    fraction_positive = np.linspace(1, 0, interpolate + 2)[1:-1]
    if interpolate:
        # TODO: This can be sped up significantly if we actually use it.
        positive_adj = positive_adj.tocsr()
        negative_adj = negative_adj.tocsr()
        mismatches = np.array(np.where((positive_adj != negative_adj).todense()))
        mismatches = mismatches[:, mismatches[0, :] <= mismatches[1, :]]
        for i in range(interpolate):
            fp = fraction_positive[i]
            idxs = random_state.choice(
                np.arange(mismatches.shape[0]), int(round(mismatches.shape[0] * fp)), replace=False)
            adj = negative_adj.tocsr()
            idx_1, idx_2 = mismatches[:, idxs]
            adj[idx_1, idx_2] = positive_adj[idx_1, idx_2]
            adj[idx_2, idx_1] = positive_adj[idx_2, idx_1]
            interpolated_adjs.append(adj.tocoo())
    return interpolated_adjs, fraction_positive.tolist()


def get_offset(length: int, random_state: Optional[np.random.RandomState] = None):
    """
    Examples:
        >>> get_offset(10, np.random.RandomState(42))
        5
    """
    if random_state is None:
        random_state = np.random.RandomState()
    min_offset = min(10, length // 2)
    offset = random_state.randint(min_offset, length - min_offset + 1)
    return offset


def get_indices(length: int,
                full_length: int,
                method: str,
                random_state: Optional[np.random.RandomState] = None):
    """
    Examples:
        >>> get_indices(3, 3, 'exact')
        (0, 3)
        >>> get_indices(3, 5, 'start')
        (0, 3)
        >>> get_indices(3, 5, 'stop')
        (2, 5)
        >>> get_indices(10, 20, 'middle', np.random.RandomState(42))
        (5, 15)
        >>> get_indices(10, 20, 'edges', np.random.RandomState(42))
        (5, 15)
    """
    assert method in ['exact', 'start', 'stop', 'middle', 'edges']

    if method in ['middle', 'edges'] and random_state is None:
        random_state = np.random.RandomState()

    gap = full_length - length

    if method == 'exact':
        assert length == full_length
        start = 0
        stop = full_length
    elif method == 'start':
        start = 0
        stop = length
    elif method == 'stop':
        start = gap
        stop = start + length
    elif method == 'middle':
        start = random_state.randint(min(10, gap // 2), max(gap - 10, gap // 2) + 1)
        stop = start + length
    elif method == 'edges':
        start = random_state.randint(min(10, gap // 2), max(length - 10, gap // 2) + 1)
        stop = full_length - (length - start)

    assert start <= stop
    return start, stop


def _combine_adjacencies(adjs: List[sparse.spmatrix], length: int):
    combined_adj = np.zeros((length, length))
    start = 0
    for adj in adjs:
        end = start + adj.shape[-1]
        combined_adj[start:end, start:end] = adj[:, :]
        start = end
    assert start == combined_adj.shape[-1]
    return combined_adj


def _split_sequence(seq: bytes, lengths: List[int]) -> List[bytes]:
    start = 0
    sequences: List[bytes] = []
    for length in lengths:
        stop = start + length
        sequences.append(seq[start:stop])
        start = stop
    assert start == len(seq)
    return sequences


def _split_adjacency(adj: sparse.spmatrix, lengths: List[int]):
    adjs = []
    start = 0
    for length in lengths:
        stop = start + length
        row = adj.row[(adj.row >= start) & (adj.row < stop)] - start
        col = adj.col[(adj.col >= start) & (adj.col < stop)] - start
        sub_adj = sparse.coo_matrix(
            (
                np.ones(len(row)),
                (row, col),
            ), shape=(length, length))
        adjs.append(sub_adj)
        start = stop
    assert start == adj.shape[-1], (start, adj.shape[-1])
    return adjs


def _permute_adjacency(adj: sparse.spmatrix, offset: int):
    row = adj.row - offset
    row = np.where(row < 0, adj.shape[0] + row, row)
    col = adj.col - offset
    col = np.where(col < 0, adj.shape[1] + col, col)
    adj_permuted = sparse.coo_matrix((adj.data, (row, col)), dtype=adj.dtype, shape=adj.shape)
    return adj_permuted


def _extract_adjacency_from_middle(start: int, stop: int, adjacency_idx_1: List[int],
                                   adjacency_idx_2: List[int]):
    """
    Examples:
        >>> _extract_adjacency_from_middle(0, 3, [0, 1, 2, 2, 3, 4], [0, 1, 2, 3, 3, 4])
        ([0, 1, 2], [0, 1, 2])
        >>> _extract_adjacency_from_middle(2, 5, [0, 1, 2, 2, 3, 4], [0, 1, 2, 3, 3, 4])
        ([0, 0, 1, 2], [0, 1, 1, 2])
    """
    keep_idx = [(pd.notnull(a) and pd.notnull(b) and start <= a < stop and start <= b < stop)
                for a, b in zip(adjacency_idx_1, adjacency_idx_2)]

    assert len(keep_idx) == len(adjacency_idx_1)
    cut_adjacency_idx_1 = [(i - start) for i, k in zip(adjacency_idx_1, keep_idx) if k]

    assert len(keep_idx) == len(adjacency_idx_2)
    cut_adjacency_idx_2 = [(i - start) for i, k in zip(adjacency_idx_2, keep_idx) if k]

    return cut_adjacency_idx_1, cut_adjacency_idx_2


def _extract_adjacency_from_edges(stop: int, start: int, adjacency_idx_1: List[int],
                                  adjacency_idx_2: List[int]):
    """
    Examples:
        >>> _extract_adjacency_from_edges(2, 4, [0, 1, 2, 2, 3, 4], [0, 1, 2, 3, 3, 4])
        ([0, 1, 2], [0, 1, 2])
    """
    keep_idx = [(pd.notnull(a) and pd.notnull(b) and (a < stop or start <= a) and
                 (b < stop or start <= b)) for a, b in zip(adjacency_idx_1, adjacency_idx_2)]

    assert len(keep_idx) == len(adjacency_idx_1)
    cut_adjacency_idx_1 = [
        (i if i < stop else (i - (start - stop))) for i, k in zip(adjacency_idx_1, keep_idx) if k
    ]

    assert len(keep_idx) == len(adjacency_idx_2)
    cut_adjacency_idx_2 = [
        (i if i < stop else (i - (start - stop))) for i, k in zip(adjacency_idx_2, keep_idx) if k
    ]

    return cut_adjacency_idx_1, cut_adjacency_idx_2
