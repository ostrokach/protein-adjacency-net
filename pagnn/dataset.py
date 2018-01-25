import operator
from typing import Callable, Generator, Iterable, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from pagnn import GAP_LENGTH, DataRow, get_adj_identity, get_adjacency, get_seq_identity

MAX_TRIES = 64
MAX_TRIES_SEQLEN = 1024


class MaxNumberOfTriesExceededError(Exception):
    pass


class DataSet(NamedTuple):
    """
    Attributes:
        seqs: List of sequences of the same length.
            If you want to include sequences of different lengths in the same batch,
            those sequences should be concatenated together.
        adjs: List of adjacency matrices.
            There should be one adjacency matrix for each *concatenated* sequence in `seqs`.
            The adjacency matrix should have three dimensions, with one or more adjacency
            matrices concatenated together.
        targets: List of target scores.
            There should be one target score for each sequence in `seqs`, for each
            *unconcatenated* sequence in *seq*, and for each Z dimension in the 3D adjacency matrix.
    """
    seqs: List[bytes]
    adjs: List[np.ndarray]
    targets: List[float]

    def _validate(self: 'DataSet'):
        # === Sequences ===
        assert self.seqs and isinstance(self.seqs, list)
        n_seq = len(self.seqs)  # x
        seq_len = len(self.seqs[0])  # z
        # All sequences in the list have the same length
        assert all(len(self.seqs[i]) == seq_len for i in range(1, n_seq))
        # === Adjacencies ===
        assert self.adjs and isinstance(self.adjs, list)
        # Correct dimensions
        assert all(len(adj.shape) == 3 for adj in self.adjs)
        assert all((adj.shape[1] == adj.shape[2]) for adj in self.adjs)
        assert all(adj.shape[0] == self.adjs[0].shape[0] for adj in self.adjs[1:])
        # Sum of adjacencies adds up to sequence length
        assert sum(adj.shape[-1] for adj in self.adjs) == seq_len
        # === Targets ===
        assert self.targets and isinstance(self.targets, list)
        assert len(self.targets) == n_seq * len(self.adjs) * self.adjs[0].shape[0]


# === Positive training examples ===


def iter_datasets(rows: Iterable[DataRow]) -> Iterator[DataSet]:
    """Convert an iterable over rows into an iterable over datasets."""
    for row in rows:
        seq = row.sequence.replace('-', '').encode('ascii')
        adj = np.array([get_adjacency(len(seq), row.adjacency_idx_1, row.adjacency_idx_2)])
        dataset = DataSet([seq], [adj], [])
        yield dataset


def iter_dataset_batches(rows: Iterable[DataRow], max_seq_len: int = 10_000) -> Iterator[DataSet]:
    """Combine several rows into a single dataset.

    Args:
        rows: List or iterator of rows.
        max_seq_len: Maximum number of residues that a sequence example can have.

    Yields:
        Inputs for machine learning.
    """
    batch: List[DataRow] = []
    batch_len = 0
    for row in rows:
        if batch_len + len(row.sequence) <= max_seq_len:
            batch.append(row)
            batch_len += len(row.sequence)
        else:
            yield _concat_rows(batch)
            batch = [row]
            batch_len = len(row.sequence)
    yield _concat_rows(batch)


def _concat_rows(rows: List[DataRow]) -> DataSet:
    """Combine one or more rows into a single dataset."""
    seqs_to_concat = [row.sequence.replace('-', '').encode('ascii') for row in rows]
    seq = (b'X' * GAP_LENGTH).join(seqs_to_concat)
    adjs = [
        np.array([get_adjacency(len(seq), row.adjacency_idx_1, row.adjacency_idx_2)])
        for seq, row in zip(seqs_to_concat, rows)
    ]
    return DataSet([seq], adjs, [])


# === Negative training examples ===


def add_negative_example(ds: DataSet,
                         method='permute',
                         interpolate: int = 0,
                         datagen: Optional[Generator[DataRow, Tuple[Callable, int], None]] = None,
                         random_state: Optional[np.random.RandomState] = None) -> DataSet:
    assert method in ['permute', 'start', 'stop', 'middle', 'edges', 'exact']
    positive_seq = ds.seqs[0]

    negative_seq = b''
    negative_adjs: List[np.ndarray] = []
    if method == 'permute':
        positive_adj = _combine_adjacencies([adj[0, :, :] for adj in ds.adjs], len(positive_seq))
        # Permute a sequence until we find something with low sequence identity
        n_tries = 0
        seq_identity = 1.0
        adj_identity = 1.0
        while seq_identity > 0.2 or adj_identity > 0.2:
            n_tries += 1
            if n_tries > MAX_TRIES:
                raise MaxNumberOfTriesExceededError()
            offset = get_offset(len(positive_seq))
            # Permute sequence
            negative_seq = positive_seq[offset:] + positive_seq[:offset]
            seq_identity = get_seq_identity(positive_seq, negative_seq)
            # Permute adjacencies
            negative_adj = _permute_adjacency(positive_adj, offset)
            adj_identity = get_adj_identity(positive_adj, negative_adj)
        negative_adjs = _split_adjacency(negative_adj, [adj.shape[-1] for adj in ds.adjs])
    else:
        subsequence_start = 0
        for adj in ds.adjs:
            sequence_length = adj.shape[-1]
            subsequence_stop = subsequence_start + sequence_length
            positive_subseq = positive_seq[subsequence_start:subsequence_stop]
            positive_adj = adj[0, :, :]
            # Find a negative sequence with low sequence identity
            n_tries = 0
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
                        row = datagen.send((operator.eq, int(sequence_length // 20 * 20)))
                        if len(row.sequence.replace('-', '')) == sequence_length:
                            break
                else:
                    row = datagen.send((operator.ge, sequence_length + 20))
                template_seq = row.sequence.replace('-', '')
                start, stop = get_indices(sequence_length, len(template_seq), method, random_state)
                if method in ['exact', 'start', 'stop', 'middle']:
                    cut_template_seq = (template_seq[start:stop]).encode('ascii')
                    cut_adjacency_idx_1, cut_adjacency_idx_2 = _extract_adjacency_from_middle(
                        start, stop, row.adjacency_idx_1, row.adjacency_idx_2)
                else:
                    cut_template_seq = (template_seq[:start] + template_seq[stop:]).encode('ascii')
                    cut_adjacency_idx_1, cut_adjacency_idx_2 = _extract_adjacency_from_edges(
                        start, stop, row.adjacency_idx_1, row.adjacency_idx_2)
                seq_identity = get_seq_identity(positive_subseq, cut_template_seq)
                negative_adj = get_adjacency(positive_adj.shape[0], cut_adjacency_idx_1,
                                             cut_adjacency_idx_2)
                adj_identity = get_adj_identity(positive_adj, negative_adj)
            negative_seq += cut_template_seq
            negative_adjs.append(negative_adj)
            subsequence_start = subsequence_stop

    assert len(negative_seq) == len(positive_seq)
    assert len(negative_adjs) == len(ds.adjs)

    targets = []
    for i, t in enumerate(ds.targets):
        targets.append(t)
        if (i + 1) % len(ds.seqs) == 0:
            targets.append(0)

    return DataSet(
        seqs=ds.seqs + [negative_seq],
        adjs=[
            np.vstack([ds.adjs[i], np.expand_dims(negative_adjs[i], 0)])
            for i in range(len(ds.adjs))
        ],
        targets=targets)


def interpolate_sequences(
        positive_seq: bytes,
        negative_seq: bytes,
        interpolate: int = 0,
        random_state: Optional[np.random.RandomState] = None) -> Tuple[List[bytes], List[float]]:
    """
    Examples:
        >>> interpolated_seqs, interpolated_targets = interpolate_sequences(
        ...     positive_seq, negative_seq, interpolate)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    interpolated_seqs = []
    fraction_positive = np.linspace(1, 0, interpolate + 2)[1:-1]
    if interpolate:
        seq_space = np.array([list(positive_seq), list(negative_seq)])
        for i in range(interpolate):
            fp = fraction_positive[i]
            # If `mask` is 1, use negative sequence:
            mask = (random_state.random(len(positive_seq)) > fp).astype(int)
            seq = bytes(seq_space[mask, range(len(positive_seq))].tolist())
            interpolated_seqs.append(seq)
    return interpolated_seqs, fraction_positive.tolist()


def interpolate_adjacencies(positive_adj: np.ndarray,
                            negative_adj: np.ndarray,
                            interpolate: int = 0,
                            random_state: Optional[np.random.RandomState] = None
                           ) -> Tuple[List[np.ndarray], List[float]]:
    """

    Examples:
        >>> interpolated_adjs, interpolated_targets = list(
        ...     zip(*[
        ...         interpolate_adjacencies(positive_adj, negative_adj, interpolate)
        ...         for positive_adj, negative_adj in zip(positive_adjs, negative_adjs)
        ...     ]))
        >>> adjs=[
        ...     np.array(adj_tuple)
        ...     for adj_tuple in zip(positive_adjs, *interpolated_adjs, negative_adjs)
        ... ]
    """
    if random_state is None:
        random_state = np.random.RandomState()
    interpolated_adjs = []
    fraction_positive = np.linspace(1, 0, interpolate + 2)[1:-1]
    if interpolate:
        mismatches = np.array(np.where(positive_adj != negative_adj))
        mismatches = mismatches[mismatches[:, 0] <= mismatches[:, 1]]
        for i in range(interpolate):
            fp = fraction_positive[i]
            idxs = random_state.choice(
                np.arange(mismatches.shape[0]), round(mismatches.shape[0] * fp), replace=False)
            adj = negative_adj.copy()
            idx_1, idx_2 = mismatches[idxs].T
            adj[idx_1, idx_2] = positive_adj[idx_1, idx_2]
            adj[idx_2, idx_1] = positive_adj[idx_2, idx_1]
            interpolated_adjs.append(adj)
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
        >>> get_indices(3, 5, 'start')
        (0, 3)
        >>> get_indices(3, 5, 'stop')
        (2, 5)
        >>> get_indices(10, 20, 'middle', np.random.RandomState(42))
        (5, 15)
        >>> get_indices(10, 20, 'edges', np.random.RandomState(42))
        (5, 15)
    """
    assert method in ['start', 'stop', 'middle', 'edges']

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
        stop = full_length - (length - stop)

    if method in ['start', 'stop', 'middle']:
        assert start <= stop
    else:
        assert stop <= start

    return start, stop


def _combine_adjacencies(adjs, length):
    combined_adj = np.zeros((length, length))
    start = 0
    for adj in adjs:
        end = start + adj.shape[-1]
        combined_adj[start:end, start:end] = adj[:, :]
        start = end
    assert start == combined_adj.shape[-1]
    return combined_adj


def _split_adjacency(adj, lengths):
    adjs = []
    start = 0
    for length in lengths:
        stop = start + length
        sub_adj = adj[start:stop, start:stop]
        adjs.append(sub_adj)
        start = stop
    assert start == adj.shape[-1]
    return adjs


def _permute_adjacency(adj, offset):
    adj_permuted = np.zeros(adj.shape)
    adj_permuted[:offset, :offset] = adj[offset:, offset:]
    adj_permuted[offset:, offset:] = adj[:offset, :offset]
    return adj


def _extract_adjacency_from_middle(start, stop, adjacency_idx_1, adjacency_idx_2):
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


def _extract_adjacency_from_edges(stop, start, adjacency_idx_1, adjacency_idx_2):
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
