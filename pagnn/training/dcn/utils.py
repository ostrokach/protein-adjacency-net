import argparse
import cProfile
import logging
import queue
from contextlib import ExitStack, contextmanager
from typing import Dict, Iterator, List, Mapping

import numpy as np
import torch.multiprocessing as mp

from pagnn import settings
from pagnn.dataset import dataset_to_gan, row_to_dataset
from pagnn.io import gen_datarows_shuffled, iter_datarows_shuffled
from pagnn.types import DataSetGAN
from pagnn.utils import (
    basic_permuted_sequence_adder,
    negative_sequence_adder,
    pc_identity_to_structure_quality,
)

from .args import Args

logger = logging.getLogger(__name__)


def prepare_dataset(positive_rowgen, negative_dsgen, args=None, random_state=None) -> DataSetGAN:
    while True:
        if args and args.permute_positives:
            assert random_state is not None
        pos_row = next(positive_rowgen)
        if args and args.predict_pc_identity:
            pos_row = pos_row._replace(target=pc_identity_to_structure_quality(pos_row.target))
        else:
            pos_row = pos_row._replace(target=1)
        pos_ds = row_to_dataset(
            pos_row, permute=args and args.permute_positives, random_state=random_state
        )
        pos_dsg = dataset_to_gan(pos_ds)
        if args and not dataset_matches_spec(pos_dsg, args):
            continue
        ds = negative_dsgen.send(pos_dsg)
        if ds is None:
            continue
        return ds


def dataset_matches_spec(ds: DataSetGAN, args: Args) -> bool:
    n_aa = ds.seqs[0].n
    if not (args.min_seq_length <= n_aa < args.max_seq_length):
        logger.debug(f"Wrong sequence length: {n_aa}.")
        return False
    row, col = ds.adjs[0].indices
    frac_interactions = float(sum(abs(row - col) > 3)) / len(row)
    if frac_interactions < 0.05:
        logger.debug(f"Too few interactions: {frac_interactions}.")
        return False
    return True


# Cache file ops


@contextmanager
def open_cachefile(cache_file_stem, mode="rb"):
    with ExitStack() as stack:
        index_fh = stack.enter_context(cache_file_stem.with_suffix(".index").open(mode))
        data_fh = stack.enter_context(cache_file_stem.with_suffix(".data").open(mode))
        yield index_fh, data_fh


def _write_ds_to_cache(index_fh, data_fh, ds_source) -> None:
    for ds in ds_source:
        data_buf = ds.to_buffer()
        size_buf = data_buf.size.to_bytes(4, "little")
        index_fh.write(size_buf)
        data_fh.write(data_buf)


def _read_ds_from_cache(index_fh, data_fh) -> Iterator[DataSetGAN]:
    while True:
        size_buf = index_fh.read(4)
        size = int.from_bytes(size_buf, "little")
        if size == 0:
            break
        data_buf = data_fh.read(size)
        ds = DataSetGAN.from_buffer(data_buf)
        yield ds


def _iter_to_completion(p, q):
    p.start()
    while True:
        try:
            ds = q.get(timeout=120)
        except queue.Empty:
            p.join()
            return
        else:
            yield ds


# Training data (generated on the fly)


def get_training_datasets(args: argparse.Namespace) -> Iterator[DataSetGAN]:
    random_state = np.random.RandomState(args.array_id)

    positive_rowgen = iter_datarows_shuffled(
        sorted(args.training_data_path.glob("database_id=*/*.parquet")),
        columns={
            "qseq": "sequence",
            "residue_idx_1_corrected": "adjacency_idx_1",
            "residue_idx_2_corrected": "adjacency_idx_2",
            "distances": None,
            "pc_identity": "target",
        },
        random_state=random_state,
    )

    if "." not in args.training_methods:
        negative_dsgen = basic_permuted_sequence_adder(
            num_sequences=args.num_negative_examples, keep_pos=True, random_state=random_state
        )
    else:
        raise NotImplementedError()
    next(negative_dsgen)

    while True:
        ds = prepare_dataset(positive_rowgen, negative_dsgen, args, random_state=random_state)
        yield ds


def get_data_pipe(args):
    ctx = mp.get_context("spawn")
    if (
        args.training_data_cache is not None
        and args.training_data_cache.with_suffix(".index").is_file()
        and args.training_data_cache.with_suffix(".data").is_file()
    ):
        logger.info("Reading training data from cache.")
        q = ctx.Queue(32768)
        if settings.PROFILER is not None:
            p = ctx.Process(target=profiled_worker, args=("read_ds_worker(args, q)", args, q))
        else:
            p = ctx.Process(target=read_ds_worker, args=(args, q))
    else:
        logger.info("Generating training data as we go.")
        q = ctx.Queue(8192)
        if settings.PROFILER is not None:
            p = ctx.Process(target=profiled_worker, args=("generate_ds_worker(args, q)", args, q))
        else:
            p = ctx.Process(target=generate_ds_worker, args=(args, q))
    yield from _iter_to_completion(p, q)


def profiled_worker(fn_call: str, args, q):
    cProfile.runctx(fn_call, globals(), locals(), filename="child.prof")


def read_ds_worker(args, q) -> None:
    with open_cachefile(args.training_data_cache) as (index_fh, data_fh):
        for ds in _read_ds_from_cache(index_fh, data_fh):
            q.put(ds)


def generate_ds_worker(args, q):
    ds_source = get_training_datasets(args)
    cache_file_stem = args.training_data_cache
    with ExitStack() as stack:
        if cache_file_stem is not None:
            logger.info("Writing generated training data to cache.")
            index_fh, data_fh = stack.enter_context(open_cachefile(cache_file_stem, "wb"))
        for i, ds in enumerate(ds_source):
            q.put(ds)
            if cache_file_stem is not None:
                _write_ds_to_cache(index_fh, data_fh, [ds])
            if args.num_sequences_to_process and (i + 1) >= args.num_sequences_to_process:
                break


# Validation data (loaded into memory)


def get_internal_validation_datasets(args: Args) -> Mapping[str, List[DataSetGAN]]:
    logger.info("Setting up validation datapipe...")
    internal_validation_datasets: Dict[str, List[DataSetGAN]] = {}
    for method in args.validation_methods.split("."):
        datagen_name = (
            f"validation_gan_{method}_{args.validation_min_seq_identity}"
            f"_{args.validation_num_sequences}"
        )
        cache_file_stem = args.validation_cache_path.joinpath(datagen_name)
        try:
            with open_cachefile(cache_file_stem) as (index_fh, data_fh):
                ds_list = list(_read_ds_from_cache(index_fh, data_fh))
            logger.info("Loaded validation datagen from cache: '%s'.", cache_file_stem)
        except FileNotFoundError:
            logger.info("Generating validation datagen: '%s'.", datagen_name)
            if args.validation_data_path is None:
                raise Exception(
                    "Cached validation data not found; `validation_data_path` must be provided!"
                )
            random_state = np.random.RandomState(sum(ord(c) for c in method))
            ds_list = _get_internal_validation_dataset(args, method, random_state)
            cache_file_stem.parent.mkdir(exist_ok=True)
            with open_cachefile(cache_file_stem, "wb") as (index_fh, data_fh):
                _write_ds_to_cache(index_fh, data_fh, ds_list)
        assert len(ds_list) == args.validation_num_sequences
        internal_validation_datasets[datagen_name] = ds_list
    return internal_validation_datasets


def _get_internal_validation_dataset(
    args: Args, method: str, random_state: np.random.RandomState
) -> List[DataSetGAN]:
    columns = {
        "qseq": "sequence",
        "residue_idx_1_corrected": "adjacency_idx_1",
        "residue_idx_2_corrected": "adjacency_idx_2",
        "distances": None,
        "pc_identity": "target",
    }
    positive_rowgen = iter_datarows_shuffled(
        sorted(args.training_data_path.glob("database_id=*/*.parquet")),
        columns=columns,
        random_state=random_state,
    )

    negative_rowgen = gen_datarows_shuffled(
        sorted(args.validation_data_path.glob("database_id=*/*.parquet")),
        columns=columns,
        random_state=random_state,
    )
    next(negative_rowgen)

    negative_dsgen = negative_sequence_adder(
        negative_rowgen, method, num_sequences=1, keep_pos=True, random_state=random_state
    )
    next(negative_dsgen)

    ds_list: List[DataSetGAN] = []
    while len(ds_list) < args.validation_num_sequences:
        ds = prepare_dataset(positive_rowgen, negative_dsgen, args, random_state)
        ds_list.append(ds)
    assert len(ds_list) == args.validation_num_sequences
    return ds_list
