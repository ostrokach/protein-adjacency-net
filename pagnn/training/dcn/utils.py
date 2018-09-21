import argparse
import logging
import math
import pickle
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable

import pagnn
from pagnn import settings
from pagnn.dataset import dataset_to_gan, row_to_dataset
from pagnn.io import gen_datarows_shuffled, iter_datarows_shuffled
from pagnn.types import DataRow, DataSetGAN, RowGen, DataSetGenM
from pagnn.utils import (
    basic_permuted_sequence_adder,
    get_rowgen_mut,
    negative_sequence_adder,
    remove_eye_sparse,
)

from .args import Args

logger = logging.getLogger(__name__)

# === Training / Validation Batches ===


def generate_batch(
    args: Args,
    net: nn.Module,
    positive_rowgen: RowGen,
    negative_ds_gen: Optional[DataSetGenM] = None,
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


# === Generators ===


def get_training_datasets(
    args: argparse.Namespace, data_path: Path, random_state=None
) -> Tuple[Iterator[DataRow], Generator[DataSetGAN, DataSetGAN, None]]:
    logger.info("Setting up training datagen...")
    rowgen_pos = iter_datarows_shuffled(
        sorted(args.training_data_path.glob("database_id=*/*.parquet")),
        columns={
            "qseq": "sequence",
            "residue_idx_1_corrected": "adjacency_idx_1",
            "residue_idx_2_corrected": "adjacency_idx_2",
        },
        random_state=random_state,
    )

    if "." not in args.training_methods:
        negative_ds_gen = basic_permuted_sequence_adder(
            num_sequences=1, keep_pos=False, random_state=random_state
        )
    else:
        raise NotImplementedError()
    next(negative_ds_gen)
    return rowgen_pos, negative_ds_gen


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
            dataset = _get_internal_validation_dataset(args, method, random_state)
            with cache_file.open("wb") as fout:
                pickle.dump(dataset, fout, pickle.HIGHEST_PROTOCOL)
        internal_validation_datasets[datagen_name] = dataset
    return internal_validation_datasets


def _get_internal_validation_dataset(
    args: Args, method: str, random_state: np.random.RandomState
) -> List[DataSetGAN]:
    columns = {
        "qseq": "sequence",
        "residue_idx_1_corrected": "adjacency_idx_1",
        "residue_idx_2_corrected": "adjacency_idx_2",
    }
    rowgen_pos = iter_datarows_shuffled(
        sorted(args.training_data_path.glob("database_id=*/*.parquet")),
        columns=columns,
        random_state=random_state,
    )

    rowgen_neg = gen_datarows_shuffled(
        sorted(args.validation_data_path.glob("database_id=*/*.parquet")),
        columns=columns,
        random_state=random_state,
    )
    next(rowgen_neg)

    nsa = negative_sequence_adder(
        rowgen_neg, method, num_sequences=1, keep_pos=True, random_state=random_state
    )
    next(nsa)

    dataset: List[DataSetGAN] = []
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
            assert ds is not None
            dataset.append(ds)
            progressbar.update(1)

    assert len(dataset) == args.validation_num_sequences
    return dataset


def get_external_validation_datasets(args, root_path, data_path) -> Mapping[str, List[DataSetGAN]]:
    external_validation_datagens: Dict[str, List[DataSetGAN]] = {}
    # for mutation_class in ['protherm', 'humsavar']:
    for mutation_class in ["protherm"]:
        external_validation_datagens[f"validation_{mutation_class}"] = _get_mutation_dataset(
            mutation_class, data_path
        )
    return external_validation_datagens


def _get_mutation_dataset(mutation_class: str, data_path: Path) -> List[DataSetGAN]:

    mutation_datarows = get_rowgen_mut(mutation_class, data_path)
    mutation_datasets = (dataset_to_gan(row_to_dataset(row, target=1)) for row in mutation_datarows)

    mutation_dsg = []
    for pos_ds in mutation_datasets:
        assert pos_ds.meta is not None
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
