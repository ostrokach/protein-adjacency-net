import itertools
import logging
import pickle
from pathlib import Path
from typing import Callable, Generator, Iterator, List, NamedTuple, Tuple

import numpy as np
import torch
import tqdm
from scipy import sparse
from torch.autograd import Variable

import pagnn
from pagnn import DataRow, DataSet, expand_adjacency, get_seq_array

logger = logging.getLogger(__name__)


class DataVar(NamedTuple):
    seq: Variable
    adj: Variable


DataGen = Callable[[], Iterator[Tuple[List[DataVar], List[DataVar]]]]


def to_numpy(array: Variable) -> np.ndarray:
    """Convert PyTorch `Variable` to numpy array."""
    if pagnn.CUDA:
        return array.data.cpu().numpy()
    else:
        return array.data.numpy()


def to_tensor(array: np.ndarray) -> torch.FloatTensor:
    return torch.FloatTensor(array)


def to_sparse_tensor(sparray: sparse.spmatrix) -> torch.sparse.FloatTensor:
    i = torch.LongTensor(np.vstack([sparray.row, sparray.col]))
    v = torch.FloatTensor(sparray.data)
    s = torch.Size(sparray.shape)
    return torch.sparse.FloatTensor(i, v, s)


def to_variable(array: np.ndarray) -> Variable:
    """Convert numpy array to PyTorch `Variable`."""
    tensor = torch.Tensor(array)
    if pagnn.CUDA:
        tensor = tensor.cuda()
    return Variable(tensor)


def dataset_to_variable(ds: DataSet) -> DataVar:
    seq = to_sparse_tensor(get_seq_array(ds.seq))
    adj = to_sparse_tensor(expand_adjacency(ds.adj))
    if pagnn.CUDA:
        seq = seq.cuda()
        adj = adj.cuda()
    datavar = DataVar(Variable(seq.to_dense().unsqueeze(0)), Variable(adj.to_dense()))
    return datavar


def get_training_datagen(working_path: Path, min_seq_identity: int) -> DataGen:
    datagen_pos = _get_datagen_pos(f'adjacency_matrix_training_gt{min_seq_identity}.parquet',
                                   working_path)
    datagen_neg = _get_datagen_neg(
        f'adjacency_matrix_training_gt{min_seq_identity}_gbseqlen.parquet', working_path)

    def training_datagen_1():
        for row in datagen_pos:
            dataset_pos = pagnn.row_to_dataset(row)
            try:
                datasets_neg = pagnn.add_negative_example(
                    [dataset_pos],
                    method='start',
                    datagen=datagen_neg,
                )
            except pagnn.MaxNumberOfTriesExceededError as e:
                logger.error("%s: %s", type(e), e)
                continue
            datavar_pos = dataset_to_variable(dataset_pos)
            datavar_neg = dataset_to_variable(datasets_neg[0])
            yield [datavar_pos], [datavar_neg]

    def training_datagen_2():
        batch_pos = []
        for i, row in enumerate(datagen_pos):
            dataset_pos = pagnn.row_to_dataset(row)
            batch_pos.append(dataset_pos)
            if (i + 1) % 100 == 0:
                batch_neg = pagnn.add_negative_example(
                    batch_pos,
                    method='permute',
                    datagen=datagen_neg,
                )
                for dataset_pos, dataset_neg in zip(batch_pos, batch_neg):
                    if len(dataset_pos.seq) < 20 or len(dataset_neg.seq) < 20:
                        continue
                    datavar_pos = dataset_to_variable(dataset_pos)
                    datavar_neg = dataset_to_variable(dataset_neg)
                    yield [datavar_pos], [datavar_neg]
                batch_pos = []

    return training_datagen_2


def get_validation_datagen(working_path: Path) -> DataGen:
    datagen_pos = _get_datagen_pos('adjacency_matrix_validation_gt80.parquet', working_path,
                                   np.random.RandomState(42))
    datagen_neg = _get_datagen_neg('adjacency_matrix_validation_gt80_gbseqlen.parquet',
                                   working_path, np.random.RandomState(42))

    datasets_cache_file = (
        working_path.joinpath('adjacency_matrix_validation_gt80-exact_dataset_cache.pickle'))
    try:
        logger.info("Loading validation dataset from file: '%s'", datasets_cache_file)
        with datasets_cache_file.open('rb') as fin:
            datasets = pickle.load(fin)
    except FileNotFoundError:
        logger.info("Loading validation dataset from file failed. Recalculating...")
        datasets = []
        for row in tqdm.tqdm(itertools.islice(datagen_pos, 1_000), total=100_000):
            dataset_pos = pagnn.row_to_dataset(row)
            try:
                datasets_neg = pagnn.add_negative_example(
                    [dataset_pos],
                    method='exact',
                    datagen=datagen_neg,
                )
            except pagnn.MaxNumberOfTriesExceededError as e:
                logger.error("%s: %s", type(e), e)
                continue
            datasets.append((dataset_pos, datasets_neg[0]))
        with datasets_cache_file.open('wb') as fout:
            pickle.dump(datasets, fout)

    def validation_datagen():
        for pos, neg in datasets:
            yield ([dataset_to_variable(pos)], [dataset_to_variable(neg)])

    return validation_datagen


def _get_datagen_pos(root_folder_name: str, working_path: Path,
                     random_state=None) -> Iterator[DataRow]:
    parquet_folders = sorted(
        working_path.parent.joinpath('threshold_by_pc_identity')
        .joinpath(root_folder_name).glob('database_id=*'))

    parquet_folder_weights_file = working_path.joinpath(
        root_folder_name.partition('.')[0] + '-weights.pickle')
    parquet_folder_weights = _load_parquet_weights(parquet_folder_weights_file, parquet_folders)
    assert len(parquet_folders) == len(parquet_folder_weights)

    datagen_pos = pagnn.iter_dataset_rows(
        parquet_folders,
        parquet_folder_weights,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
        },
        random_state=random_state)

    return datagen_pos


def _get_datagen_neg(root_folder_name: str, working_path: Path,
                     random_state=None) -> Generator[DataRow, None, None]:
    parquet_folders = sorted(
        working_path.parent.joinpath('group_by_sequence_length')
        .joinpath(root_folder_name).glob('qseq_length_bin=*'))

    parquet_folder_weights_file = working_path.joinpath(
        root_folder_name.partition('.')[0] + '-weights.pickle')
    parquet_folder_weights = _load_parquet_weights(parquet_folder_weights_file, parquet_folders)
    assert len(parquet_folders) == len(parquet_folder_weights)

    datagen_neg = pagnn.iter_dataset_rows(
        parquet_folders,
        parquet_folder_weights,
        columns={
            'qseq': 'sequence',
            'residue_idx_1_corrected': 'adjacency_idx_1',
            'residue_idx_2_corrected': 'adjacency_idx_2',
        },
        seq_length_constraint=True,
        random_state=random_state)
    next(datagen_neg)

    return datagen_neg


def _load_parquet_weights(filepath: Path, parquet_folders: List[Path]) -> np.ndarray:
    try:
        logger.info("Loading folder weights from file: '%s'", filepath)
        with filepath.open('rb') as fin:
            d = pickle.load(fin)
            parquet_folder_weights = np.array([d[p.name] for p in parquet_folders])
    except FileNotFoundError:
        logger.info("Loading folder weights from file failed! Recomputing...")
        parquet_folder_weights = pagnn.get_weights(parquet_folders)
        d = {p.name: w for p, w in zip(parquet_folders, parquet_folder_weights)}
        with filepath.open('wb') as fout:
            pickle.dump(d, fout, pickle.HIGHEST_PROTOCOL)
    return parquet_folder_weights
