"""Functions for generating and manipulating DataSets."""
import operator
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from pagnn import utils
from pagnn.exc import MaxNumberOfTriesExceededError, SequenceTooLongError
from pagnn.types import DataRow, DataSet

MAX_TRIES = 1024
MAX_TRIES_SEQLEN = 8192

# === Positive training examples ===


def row_to_dataset(row: DataRow, target: float) -> DataSet:
    """Convert a `DataRow` into a `DataSet`."""
    seq = row.sequence.replace('-', '').encode('ascii')
    adj = utils.get_adjacency(len(seq), row.adjacency_idx_1, row.adjacency_idx_2)
    known_fields = {'Index', 'sequence', 'adjacency_idx_1', 'adjacency_idx_2', 'target'}
    if set(row._fields) - set(known_fields):
        meta = {k: v for k, v in row._asdict().items() if k not in known_fields}
    else:
        meta = None
    return DataSet(seq, adj, target, meta)


# === Negative training examples ===


def get_negative_example(ds: DataSet,
                         method: str,
                         datagen: Generator[DataRow, Tuple[Callable, int], None],
                         random_state: Optional[np.random.RandomState] = None) -> DataSet:
    """Find a valid negative control for a given `ds`.

    Raises:
        MaxNumberOfTriesExceededError
    """
    assert method in ['permute', 'exact', 'start', 'stop', 'middle', 'edges'], method
    n_tries = 0
    seq_identity = 1.0
    adj_identity = 1.0
    while seq_identity > 0.2 or adj_identity > 0.3:
        n_tries += 1
        if n_tries > MAX_TRIES:
            raise MaxNumberOfTriesExceededError(n_tries)
        if method == 'permute':
            offset = get_offset(len(ds.seq))
            negative_seq = ds.seq[offset:] + ds.seq[:offset]
            negative_adj = None
            seq_identity = utils.get_seq_identity(ds.seq, negative_seq)
            adj_identity = 0
            break
        elif method == 'exact':
            n_tries_seqlen = 0
            while True:
                n_tries_seqlen += 1
                if n_tries_seqlen > MAX_TRIES_SEQLEN:
                    raise MaxNumberOfTriesExceededError(n_tries_seqlen)
                row = datagen.send((operator.eq, int(len(ds.seq) // 20 * 20)))
                if row is None:
                    raise SequenceTooLongError(
                        f"Could not find a generator for target_seq_length: {len(ds.seq)}.")
                negative_ds = row_to_dataset(row, target=0)
                if len(negative_ds.seq) == len(ds.seq):
                    break
        else:
            row = datagen.send((operator.ge, len(ds.seq) + 20))
            if row is None:
                raise SequenceTooLongError(
                    f"Could not find a generator for target_seq_length: {len(ds.seq)}.")
            negative_ds = row_to_dataset(row, target=0)
        start, stop = get_indices(len(ds.seq), len(negative_ds.seq), method, random_state)
        if method not in ['edges']:
            negative_seq = negative_ds.seq[start:stop]
            negative_adj = _extract_adjacency_from_middle(start, stop, negative_ds.adj)
        else:
            negative_seq = negative_ds.seq[:start] + negative_ds.seq[stop:]
            negative_adj = _extract_adjacency_from_edges(start, stop, negative_ds.adj)
        seq_identity = utils.get_seq_identity(ds.seq, negative_seq)
        adj_identity = utils.get_adj_identity(ds.adj, negative_adj)

    negative_dataset = DataSet(negative_seq, negative_adj, 0)
    return negative_dataset


def get_permuted_examples(datasets: List[DataSet],
                          random_state: Optional[np.random.RandomState] = None) -> List[DataSet]:
    """
    Generate negative examples by permuting a list of sequences together.

    Notes:
        * Makes no sense to permute the combined adjacency matrix because the resulting
          split adjacencies will have fewer contacts in total.
    """
    negative_datasets: List[DataSet] = []

    combined_seq = b''.join(ds.seq for ds in datasets)
    # combined_adj = sparse.block_diag([ds.adj for ds in datasets])
    # Permute a sequence until we find something with low sequence identity
    n_tries = 0
    seq_identity = 1.0
    while seq_identity > 0.2:
        n_tries += 1
        if n_tries > MAX_TRIES:
            raise MaxNumberOfTriesExceededError(n_tries)
        offset = get_offset(len(combined_seq))
        # Permute sequence
        negative_seq = combined_seq[offset:] + combined_seq[:offset]
        seq_identity = utils.get_seq_identity(combined_seq, negative_seq)
    negative_seqs = _split_sequence(negative_seq, [len(ds.seq) for ds in datasets])
    for nseq in negative_seqs:
        negative_datasets.append(DataSet(nseq, sparse.coo_matrix([]), 0))

    assert len(negative_datasets) == len(datasets)

    return negative_datasets


def get_offset(length: int, random_state: Optional[np.random.RandomState] = None):
    """Chose a random offset (useful for permutations, etc).

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
    Get `start` and `stop` indices for a given slice method.

    Examples:
        >>> get_indices(3, 3, 'exact')
        (0, 3)
        >>> get_indices(3, 5, 'start')
        (0, 3)
        >>> get_indices(3, 5, 'stop')
        (2, 5)
        >>> get_indices(20, 30, 'middle', np.random.RandomState(42))
        (5, 25)
        >>> get_indices(20, 30, 'edges', np.random.RandomState(42))
        (8, 18)
    """
    assert method in ['exact', 'start', 'stop', 'middle', 'edges'], method

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
        assert length >= 20
        start = random_state.randint(min(10, gap // 2), max(gap - 10, gap // 2) + 1)
        stop = start + length
    elif method == 'edges':
        assert length >= 20
        start = random_state.randint(min(10, gap // 2), min(length - 10, gap) + 1)
        stop = full_length - (length - start)

    assert start >= 0, (length, full_length, method, start, stop)
    assert stop <= full_length, (length, full_length, method, start, stop)
    assert start <= stop, (length, full_length, method, start, stop)

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
    """
    .. warning::
        Some of the contacts may be lost when the adjacency is divided into subadjacencies.
    """
    adjs = []
    start = 0
    for length in lengths:
        stop = start + length
        valid_idx = ((adj.row >= start) & (adj.row < stop) & (adj.col >= start) & (adj.col < stop))
        row = adj.row[valid_idx] - start
        col = adj.col[valid_idx] - start
        assert len(row) == len(col)
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


def _extract_adjacency_from_middle(start: int, stop: int, adj: sparse.spmatrix):
    """
    Examples:
        >>> from scipy import sparse
        >>> adj = sparse.coo_matrix((
        ...     np.ones(6), (np.array([0, 1, 2, 2, 3, 4]), np.array([0, 1, 2, 3, 3, 4])), ))
        >>> negative_adj = _extract_adjacency_from_middle(0, 3, adj)
        >>> negative_adj.row
        array([0, 1, 2], dtype=int32)
        >>> negative_adj.col
        array([0, 1, 2], dtype=int32)
        >>> negative_adj = _extract_adjacency_from_middle(2, 5, adj)
        >>> negative_adj.row
        array([0, 0, 1, 2], dtype=int32)
        >>> negative_adj.col
        array([0, 1, 1, 2], dtype=int32)
    """
    keep_idx = [(pd.notnull(a) and pd.notnull(b) and start <= a < stop and start <= b < stop)
                for a, b in zip(adj.row, adj.col)]
    new_row = adj.row[keep_idx] - start
    new_col = adj.col[keep_idx] - start
    new_adj = sparse.coo_matrix(
        (adj.data[keep_idx], (new_row, new_col)),
        dtype=adj.dtype,
        shape=((stop - start, stop - start)))
    return new_adj


def _extract_adjacency_from_edges(start: int, stop: int, adj: sparse.spmatrix):
    """
    Examples:
        >>> from scipy import sparse
        >>> adj = sparse.coo_matrix((
        ...     np.ones(6), (np.array([0, 1, 2, 2, 3, 4]), np.array([0, 1, 2, 3, 3, 4])), ))
        >>> negative_adj = _extract_adjacency_from_edges(2, 4, adj)
        >>> negative_adj.row
        array([0, 1, 2], dtype=int32)
        >>> negative_adj.col
        array([0, 1, 2], dtype=int32)
    """
    keep_idx = [(pd.notnull(a) and pd.notnull(b) and (a < start or stop <= a) and
                 (b < start or stop <= b)) for a, b in zip(adj.row, adj.col)]

    keep_idx = ((adj.row < start) | (adj.row >= stop)) & ((adj.col < start) | (adj.col >= stop))
    new_row = adj.row[keep_idx]
    new_row = np.where(new_row < start, new_row, new_row - (stop - start))

    new_col = adj.col[keep_idx]
    new_col = np.where(new_col < stop, new_col, new_col - (stop - start))
    try:
        new_adj = sparse.coo_matrix(
            (adj.data[keep_idx], (new_row, new_col)),
            dtype=adj.dtype,
            shape=((adj.shape[0] - (stop - start), adj.shape[1] - (stop - start))))
    except ValueError as e:
        import pdb
        pdb.set_trace()
    return new_adj
