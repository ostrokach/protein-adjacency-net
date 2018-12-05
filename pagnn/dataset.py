"""Functions for generating and manipulating DataSets."""
import enum
import operator
from typing import List, Optional, Union

import numpy as np
import torch
from scipy import sparse

from pagnn import utils
from pagnn.exc import MaxNumberOfTriesExceededError, SequenceTooLongError
from pagnn.types import DataRow, DataSet, DataSetGAN, RowGenF, SparseMat
from pagnn.utils import seq_to_array

MAX_TRIES = 256


def row_to_dataset(
    row: DataRow,
    target: Optional[float] = None,
    permute=False,
    random_state: Optional[np.random.RandomState] = None,
) -> DataSet:
    """Convert a :any:`DataRow` into a :any:`DataSet`.

    Args:
        row: A row of data.
        target: The target associated with the row of data.
        permute: Whether or not the sequence and adjacency should be permutted by a random amount.
        random_state: RandomState for generating permutations.

    Returns:
        A DataSet containing the provided data.
    """
    seq = seq_to_array(row.sequence.replace("-", "").encode("ascii"))
    adj = utils.get_adjacency(seq.shape[1], row.adjacency_idx_1, row.adjacency_idx_2)
    target = target if target is not None else row.target
    if permute:
        assert random_state is not None
        seq_length = seq.size(1)
        offset = get_offset(seq_length, random_state)
        seq = permute_sequence(seq, offset)
        adj = permute_adjacency(adj, offset)
    known_fields = {"Index", "sequence", "adjacency_idx_1", "adjacency_idx_2", "target"}
    if set(row._fields) - set(known_fields):
        meta = {k: v for k, v in row._asdict().items() if k not in known_fields}
    else:
        meta = {}
    return DataSet(seq, adj, target, meta)


def dataset_to_gan(ds: DataSet) -> DataSetGAN:
    """Convert a :any:`DataSet` into a :any:`DataSetGAN`."""
    adj_indices = torch.stack(
        [torch.tensor(ds.adj.row, dtype=torch.long), torch.tensor(ds.adj.col, dtype=torch.long)]
    )
    adj_values = torch.tensor(ds.adj.data, dtype=torch.float)
    return DataSetGAN(
        [SparseMat(ds.seq._indices(), ds.seq._values(), ds.seq.shape[0], ds.seq.shape[1])],
        [SparseMat(adj_indices, adj_values, ds.adj.shape[0], ds.adj.shape[1])],
        torch.tensor([ds.target], dtype=torch.float),
        ds.meta,
    )


@enum.unique
class Method(enum.Enum):
    """Possible ways of adding negative training examples."""

    PERMUTE = "permute"
    EXACT = "exact"
    START = "start"
    STOP = "stop"
    MIDDLE = "middle"
    EDGES = "edges"


def get_negative_example(
    dsg: DataSetGAN,
    method: Union[str, Method],
    rowgen: RowGenF,
    random_state: Optional[np.random.RandomState] = None,
) -> DataSetGAN:
    """Find a valid negative control for a given `ds`.

    Raises:
        - MaxNumberOfTriesExceededError
        - SequenceTooLongError
    """
    if isinstance(method, str):
        method = Method(method)
    assert method in Method
    n_tries = 0
    seq_identity = 1.0
    adj_identity = 1.0
    seq_length = dsg.seqs[0].n
    while seq_identity > 0.2 or adj_identity > 0.3:
        n_tries += 1
        if n_tries > MAX_TRIES:
            raise MaxNumberOfTriesExceededError(n_tries)
        if method == Method.PERMUTE:
            offset = get_offset(seq_length, random_state)
            negative_seq = SparseMat(
                torch.cat([dsg.seqs[0].indices[:, offset:], dsg.seqs[0].indices[:, :offset]], 1),
                dsg.seqs[0].values,
                dsg.seqs[0].m,
                dsg.seqs[0].n,
            )
            negative_adj = None
            seq_identity = utils.get_seq_identity(dsg.seqs[0], negative_seq)
            adj_identity = 0
        else:
            row = None
            op = operator.eq if method == Method.EXACT else operator.ge
            while row is None and n_tries < MAX_TRIES:
                n_tries += 1
                row = rowgen.send(
                    lambda df: df[op(df["sequence"].str.replace("-", "").str.len(), seq_length)]
                )
            if row is None:
                raise SequenceTooLongError(
                    f"""Could not find a generator for target_seq_length: {seq_length}."""
                )
            negative_seq, negative_adj = _row_to_seqadj(row, seq_length, method, random_state)
            seq_identity = utils.get_seq_identity(dsg.seqs[0], negative_seq)
            adj_identity = utils.get_adj_identity(dsg.adjs[0], negative_adj)

    negative_dataset = DataSetGAN([negative_seq], [negative_adj], 0)
    return negative_dataset


def _row_to_seqadj(row, seq_length, method, random_state):
    negative_ds = row_to_dataset(row, target=0)
    start, stop = get_indices(seq_length, negative_ds.seq.shape[1], method, random_state)
    if method not in [Method.EDGES]:
        negative_seq = SparseMat(
            negative_ds.seq._indices()[:, start:stop],
            negative_ds.seq._values()[start:stop],
            20,
            seq_length,
        )
        negative_adj = extract_adjacency_from_middle(start, stop, negative_ds.adj)
    else:
        negative_seq = SparseMat(
            torch.cat(
                [negative_ds.seq._indices()[:, :start], negative_ds.seq._indices()[:, stop:]], 1
            ),
            torch.cat([negative_ds.seq._values()[:start], negative_ds.seq._values()[stop:]]),
            20,
            seq_length,
        )
        negative_adj = extract_adjacency_from_edges(start, stop, negative_ds.adj)
    return negative_seq, negative_adj


def get_permuted_examples(
    datasets: List[DataSet], random_state: Optional[np.random.RandomState] = None
) -> List[DataSet]:
    """
    Generate negative examples by permuting a list of sequences together.

    Notes:
        * Makes no sense to permute the combined adjacency matrix because the resulting
          split adjacencies will have fewer contacts in total.
    """
    negative_datasets: List[DataSet] = []

    combined_seq = b"".join(ds.seq for ds in datasets)
    # combined_adj = sparse.block_diag([ds.adj for ds in datasets])
    # Permute a sequence until we find something with low sequence identity
    n_tries = 0
    seq_identity = 1.0
    while seq_identity > 0.2:
        n_tries += 1
        if n_tries > MAX_TRIES:
            raise MaxNumberOfTriesExceededError(n_tries)
        offset = get_offset(len(combined_seq), random_state)
        # Permute sequence
        negative_seq = combined_seq[offset:] + combined_seq[:offset]
        seq_identity = utils.get_seq_identity(combined_seq, negative_seq)
    negative_seqs = _split_sequence(negative_seq, [len(ds.seq) for ds in datasets])
    for nseq in negative_seqs:
        negative_datasets.append(DataSet(nseq, sparse.coo_matrix([]), 0))

    assert len(negative_datasets) == len(datasets)

    return negative_datasets


def get_offset(length: int, random_state: np.random.RandomState):
    """Chose a random offset (useful for permutations, etc).

    Examples:
        >>> get_offset(10, np.random.RandomState(42))
        5
    """
    min_offset = min(10, length // 2)
    offset = random_state.randint(min_offset, length - min_offset + 1)
    return offset


def get_indices(
    length: int, full_length: int, method: Union[str, Method], random_state: np.random.RandomState
):
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
    if isinstance(method, str):
        method = Method(method)
    assert method in Method

    if method in ["middle", "edges"] and random_state is None:
        random_state = np.random.RandomState()

    gap = full_length - length

    if method == Method.EXACT:
        assert length == full_length
        start = 0
        stop = full_length
    elif method == Method.START:
        start = 0
        stop = length
    elif method == Method.STOP:
        start = gap
        stop = start + length
    elif method == Method.MIDDLE:
        assert length >= 20
        start = random_state.randint(min(10, gap // 2), max(gap - 10, gap // 2) + 1)
        stop = start + length
    elif method == Method.EDGES:
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
        valid_idx = (adj.row >= start) & (adj.row < stop) & (adj.col >= start) & (adj.col < stop)
        row = adj.row[valid_idx] - start
        col = adj.col[valid_idx] - start
        assert len(row) == len(col)
        sub_adj = sparse.coo_matrix((np.ones(len(row)), (row, col)), shape=(length, length))
        adjs.append(sub_adj)
        start = stop
    assert start == adj.shape[-1], (start, adj.shape[-1])
    return adjs


def permute_sequence(seq: torch.sparse.FloatTensor, offset: int):
    row, col = seq._indices()
    values = seq._values()
    row_new = torch.cat([row[3:], row[:3]])
    col_new = col
    indices_new = torch.stack([row_new, col_new], 0)
    values_new = torch.cat([values[offset:], values[:offset]])
    seq_new = torch.sparse_coo_tensor(
        indices_new, values_new, size=seq.size(), dtype=seq.dtype, device=seq.device
    )
    return seq_new


def permute_adjacency(adj: sparse.spmatrix, offset: int):
    row = adj.row - offset
    row = np.where(row < 0, adj.shape[0] + row, row)
    col = adj.col - offset
    col = np.where(col < 0, adj.shape[1] + col, col)
    adj_permuted = sparse.coo_matrix((adj.data, (row, col)), dtype=adj.dtype, shape=adj.shape)
    return adj_permuted


def extract_adjacency_from_middle(start: int, stop: int, adj: sparse.spmatrix) -> SparseMat:
    """Extract adjacency matrix from ``[start..stop)`` of `adj`.

    Examples:
        >>> from scipy import sparse
        >>> adj = sparse.coo_matrix((
        ...     np.ones(6), (np.array([0, 1, 2, 2, 3, 4]), np.array([0, 1, 2, 3, 3, 4])), ))
        >>> negative_adj = extract_adjacency_from_middle(0, 3, adj)
        >>> negative_adj.row
        array([0, 1, 2], dtype=int32)
        >>> negative_adj.col
        array([0, 1, 2], dtype=int32)
        >>> negative_adj = extract_adjacency_from_middle(2, 5, adj)
        >>> negative_adj.row
        array([0, 0, 1, 2], dtype=int32)
        >>> negative_adj.col
        array([0, 1, 1, 2], dtype=int32)
    """
    keep_idx = (
        np.isfinite(adj.row)
        & np.isfinite(adj.col)
        & (start <= adj.row)
        & (adj.row < stop)
        & (start <= adj.col)
        & (adj.col < stop)
    )
    new_row = adj.row[keep_idx] - start
    new_col = adj.col[keep_idx] - start
    new_adj = SparseMat(
        torch.stack(
            [torch.tensor(new_row, dtype=torch.long), torch.tensor(new_col, dtype=torch.long)]
        ),
        torch.tensor(adj.data[keep_idx], dtype=torch.float),
        stop - start,
        stop - start,
    )
    return new_adj


def extract_adjacency_from_edges(start: int, stop: int, adj: sparse.spmatrix) -> SparseMat:
    """Extract adjacency matrix from ``[0, start)`` and from ``[stop, end)`` of ``adj``.

    Examples:
        >>> from scipy import sparse
        >>> adj = sparse.coo_matrix((
        ...     np.ones(6), (np.array([0, 1, 2, 2, 3, 4]), np.array([0, 1, 2, 3, 3, 4])), ))
        >>> negative_adj = extract_adjacency_from_edges(2, 4, adj)
        >>> negative_adj.row
        array([0, 1, 2], dtype=int32)
        >>> negative_adj.col
        array([0, 1, 2], dtype=int32)
    """
    keep_idx = (
        np.isfinite(adj.row)
        & np.isfinite(adj.col)
        & ((adj.row < start) | (adj.row >= stop))
        & ((adj.col < start) | (adj.col >= stop))
    )
    new_row = adj.row[keep_idx]
    new_row = np.where(new_row < start, new_row, new_row - (stop - start))

    new_col = adj.col[keep_idx]
    new_col = np.where(new_col < stop, new_col, new_col - (stop - start))
    new_adj = SparseMat(
        torch.stack(
            [torch.tensor(new_row, dtype=torch.long), torch.tensor(new_col, dtype=torch.long)]
        ),
        torch.tensor(adj.data[keep_idx], dtype=torch.float),
        adj.shape[0] - (stop - start),
        adj.shape[1] - (stop - start),
    )
    return new_adj
