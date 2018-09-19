import argparse
import itertools
import logging
import math
import pickle
from pathlib import Path
from typing import Callable, Dict, Generator, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable

import pagnn
from pagnn import exc, settings
from pagnn.dataset import get_negative_example, get_offset, row_to_dataset, dataset_to_gan
from pagnn.datavargan import datasets_to_datavar
from pagnn.training.common import get_rowgen_mut, get_rowgen
from pagnn.training.gan import Args
from pagnn.types import DataRow, DataSet, DataSetGAN
from pagnn.utils import array_to_seq, remove_eye_sparse

logger = logging.getLogger(__name__)

RowGen = Generator[DataRow, Tuple[Callable, int], None]

# === Training / Validation Batches ===


def get_designed_seqs(args, dataset, net_d, net_g):
    # TODO: Figure out why this is here
    seq_wt = dataset.seqs[0].decode()

    designed_seqs = []
    noise = torch.FloatTensor(args.batch_size, 256)
    if settings.CUDA:
        noise = noise.cuda()
    # for offset in range(0, max(1, math.ceil(len(seq_wt) / 128) * 128 - len(seq_wt))):
    for offset in range(0, 1):
        start = offset
        stop = start + len(seq_wt)
        datavar = datasets_to_datavar(dataset, offset=offset)
        noisev = Variable(noise.normal_(0, 1))
        pred = net_g(noisev, datavar.adjs, net_d)
        pred_np = pagnn.to_numpy(pred)
        pred_np = pred_np[:, :, start:stop]
        designed_seqs.extend([array_to_seq(pred_np[i]) for i in range(pred_np.shape[0])])
    return designed_seqs


def generate_batch(
    args: Args, net: nn.Module, positive_rowgen: RowGen, negative_ds_gen: Optional[RowGen] = None
):
    """Generate a positive and a negative dataset batch."""
    pos_seq_list = []
    neg_seq_list = []
    adjs = []
    seq_len = 0
    # TODO: 128 comes from the fact that we tested with sequences 64-256 AA in length
    while seq_len < (args.batch_size * 128):
        pos_row = next(positive_rowgen)
        pos_ds = dataset_to_gan(row_to_dataset(pos_row, 1))
        # Filter out bad datasets
        n_aa = len(pos_ds.seqs[0])
        if not (args.min_seq_length <= n_aa < args.max_seq_length):
            logger.debug(f"Skipping because wrong sequence length: {n_aa}.")
            continue
        adj_nodiag = pagnn.utils.remove_eye_sparse(pos_ds.adjs[0], 3)
        n_interactions = adj_nodiag.nnz
        if n_interactions <= 0:
            logger.debug(f"Skipping because too few interactions: {n_interactions}.")
            continue
        # Continue
        pos_dv = net.dataset_to_datavar(pos_ds)
        pos_seq_list.append(pos_dv.seqs)
        adjs.append(pos_dv.adjs)
        if negative_ds_gen is not None:
            neg_ds = negative_ds_gen.send(pos_ds)
            neg_dv = net.dataset_to_datavar(neg_ds)
            neg_seq_list.append(neg_dv.seqs)
        seq_len += pos_dv.seqs.shape[2]
    pos_seq = Variable(torch.cat([s.data for s in pos_seq_list], 2))
    assert pos_seq.shape[2] == sum(adj[0].shape[1] for adj in adjs)
    if negative_ds_gen is not None:
        neg_seq = Variable(torch.cat([s.data for s in neg_seq_list], 2))
        assert neg_seq.shape[2] == sum(adj[0].shape[1] for adj in adjs)
    else:
        neg_seq = None
    return pos_seq, neg_seq, adjs


def generate_noise(net_g, adjs):
    num_aa_out = sum(adj[net_g.n_layers].shape[1] for adj in adjs)
    noise_length = math.ceil(num_aa_out * net_g.bottleneck_features / 2048)
    if settings.CUDA:
        noise = torch.cuda.FloatTensor(1, net_g.bottleneck_size, noise_length)
    else:
        noise = torch.FloatTensor(1, net_g.bottleneck_size, noise_length)
    return noise


# === Generators ===


def get_training_datasets(
    args: argparse.Namespace, data_path: Path, random_state=None
) -> Tuple[Iterator[DataRow], Generator[DataSetGAN, DataSetGAN, None]]:
    logger.info("Setting up training datagen...")
    positive_rowgen = get_rowgen(data_path, filterable=False, random_state=random_state)
    if "." not in args.training_methods:
        negative_ds_gen = basic_permuted_sequence_adder(
            num_sequences=1, keep_pos=False, random_state=random_state
        )
    else:
        raise NotImplementedError()
    next(negative_ds_gen)
    return positive_rowgen, negative_ds_gen


def get_internal_validation_datasets(
    args: Args, validation_data_path: Optional[Path]
) -> Mapping[str, List[DataSetGAN]]:
    logger.info("Setting up validation datagen...")
    internal_validation_datasets: Dict[str, List[DataSetGAN]] = {}
    for method in args.validation_methods.split("."):
        datagen_name = (
            f"validation_gan_{method}_{args.validation_min_seq_identity}"
            f"_{args.validation_num_sequences}"
        )
        cache_file = Path(__file__).resolve().parent.joinpath("data", datagen_name + ".pickle")
        try:
            with cache_file.open("rb") as fin:
                dataset = pickle.load(fin)
            assert len(dataset) == args.validation_num_sequences
            logger.info("Loaded validation datagen from file: '%s'.", cache_file)
        except FileNotFoundError:
            logger.info("Generating validation datagen: '%s'.", datagen_name)
            if validation_data_path is None:
                raise Exception(
                    "Cached validation data not found; `validation_data_path` must be provided!"
                )
            random_state = np.random.RandomState(sum(ord(c) for c in method))
            dataset = get_validation_dataset(args, method, validation_data_path, random_state)
            with cache_file.open("wb") as fout:
                pickle.dump(dataset, fout, pickle.HIGHEST_PROTOCOL)
        internal_validation_datasets[datagen_name] = dataset
    return internal_validation_datasets


def get_external_validation_datasets(args, root_path, data_path) -> Mapping[str, List[DataSetGAN]]:
    external_validation_datagens: Dict[str, List[DataSetGAN]] = {}
    # for mutation_class in ['protherm', 'humsavar']:
    for mutation_class in ["protherm"]:
        external_validation_datagens[f"validation_{mutation_class}"] = get_mutation_dataset(
            mutation_class, data_path
        )
    return external_validation_datagens


# === Dataset loaders ===


def get_validation_dataset(
    args: argparse.Namespace, method: str, data_path: Path, random_state: np.random.RandomState
) -> List[DataSetGAN]:
    rowgen_pos = get_rowgen(
        data_path,
        # The validation dataset should already be filtered to >= 80% sequence identity
        # filters=[lambda df: df[df["pc_identity"] >= 80]],
        filterable=False,
        extra_columns=["pc_identity"],
        random_state=random_state,
    )
    rowgen_neg = get_rowgen(
        data_path,
        # The validation dataset should already be filtered to >= 80% sequence identity
        # filters=[lambda df: df[df["pc_identity"] >= 80]],
        filterable=True,
        extra_columns=["pc_identity"],
        random_state=random_state,
    )
    nsa = negative_sequence_adder(
        rowgen_neg, method, num_sequences=1, keep_pos=True, random_state=random_state
    )
    next(nsa)

    dataset = []
    with tqdm.tqdm(
        total=args.validation_num_sequences, desc=method, disable=not settings.SHOW_PROGRESSBAR
    ) as progressbar:
        while len(dataset) < args.validation_num_sequences:
            pos_row = next(rowgen_pos)
            pos_ds = dataset_to_gan(row_to_dataset(pos_row, 1))
            # Filter out bad datasets
            n_aa = len(pos_ds.seqs[0])
            if not (args.min_seq_length <= n_aa < args.max_seq_length):
                logger.debug(f"Skipping because wrong sequence length: {n_aa}.")
                continue
            adj_nodiag = remove_eye_sparse(pos_ds.adjs[0], 3)
            n_interactions = adj_nodiag.nnz
            if n_interactions <= 0:
                logger.debug(f"Skipping because too few interactions: {n_interactions}.")
                continue
            #
            ds = nsa.send(pos_ds)
            dataset.append(ds)
            progressbar.update(1)

    assert len(dataset) == args.validation_num_sequences
    return dataset


def get_mutation_dataset(mutation_class: str, data_path: Path) -> List[DataSetGAN]:

    mutation_datarows = get_rowgen_mut(mutation_class, data_path)
    mutation_datasets = (dataset_to_gan(row_to_dataset(row, target=1)) for row in mutation_datarows)

    mutation_dsg = []
    for pos_ds in mutation_datasets:
        neg_seq = bytearray(pos_ds.seqs[0])
        mutation = pos_ds.meta["mutation"]
        mutation_idx = int(mutation[1:-1]) - 1
        assert neg_seq[mutation_idx] == ord(mutation[0]), (chr(neg_seq[mutation_idx]), mutation[0])
        neg_seq[mutation_idx] = ord(mutation[-1])
        ds = pos_ds._replace(
            seqs=pos_ds.seqs + [neg_seq], targets=pos_ds.targets + [pos_ds.meta["score"]]
        )
        mutation_dsg.append(ds)

    return mutation_dsg


# === Negative dataset generators ===


def basic_permuted_sequence_adder(
    num_sequences: int, keep_pos: bool, random_state: Optional[np.random.RandomState] = None
):
    if random_state is None:
        random_state = np.random.RandomState()

    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        seq = dsg.seqs[0]
        negative_seqs = []
        for _ in range(num_sequences):
            offset = get_offset(len(seq), random_state)
            negative_seq = seq[offset:] + seq[:offset]
            negative_seqs.append(negative_seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            adjs=(dsg.adjs if keep_pos else []) + dsg.adjs * len(negative_seqs),
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )


def buffered_permuted_sequence_adder(
    rowgen: RowGen,
    num_sequences: int,
    keep_pos: bool = False,
    random_state: Optional[np.random.RandomState] = None,
) -> Generator[DataSetGAN, DataSetGAN, None]:
    """

    Args:
        rowgen: Used for **pre-populating** the generator only!
        num_sequences: Number of sequences to generate in each iteration.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    seq_buffer = [row_to_dataset(r, 0).seq for r in itertools.islice(rowgen, 512)]
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        seq = dsg.seqs[0]
        negative_seq_big = b"".join(seq_buffer)
        negative_seqs = []
        for _ in range(num_sequences):
            offset = random_state.randint(0, len(negative_seq_big) - len(seq))
            negative_seq = (negative_seq_big[offset:] + negative_seq_big[:offset])[: len(seq)]
            negative_seqs.append(negative_seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )
        # Reshuffle negative sequences
        seq_buffer.append(seq)
        random_state.shuffle(seq_buffer)
        random_state.pop()


def negative_sequence_adder(
    rowgen: RowGen,
    method: str,
    num_sequences: int,
    keep_pos: bool = False,
    random_state: Optional[np.random.RandomState] = None,
) -> Generator[DataSetGAN, DataSetGAN, None]:
    """

    Args:
        rowgen: Generator used for fetching negative rows.
        method: Method by which longer negative sequences get cut to the correct size.
        num_sequences: Number of sequences to generate in each iteration.
    """
    MAX_TRIES = 5
    if random_state is None:
        random_state = np.random.RandomState()
    negative_dsg = None
    while True:
        dsg = yield negative_dsg
        ds = DataSet(dsg.seqs[0], dsg.adjs[0], dsg.targets[0])
        negative_seqs = []
        n_tries = 0
        while len(negative_seqs) < num_sequences:
            try:
                negative_ds = get_negative_example(ds, method, rowgen, random_state)
            except (exc.MaxNumberOfTriesExceededError, exc.SequenceTooLongError) as e:
                logger.error("Encountered error '%s' for dataset '%s'", e, ds)
                if n_tries < MAX_TRIES:
                    n_tries += 1
                    continue
                else:
                    raise
            negative_seqs.append(negative_ds.seq)
        negative_dsg = dsg._replace(
            seqs=(dsg.seqs if keep_pos else []) + negative_seqs,
            targets=(dsg.targets if keep_pos else []) + [0] * num_sequences,
        )
