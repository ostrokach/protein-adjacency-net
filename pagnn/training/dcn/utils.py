import argparse
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch

from pagnn.datapipe import set_buf_size
from pagnn.dataset import dataset_to_gan, row_to_dataset
from pagnn.io import gen_datarows_shuffled, iter_datarows_shuffled
from pagnn.types import DataSetGAN
from pagnn.utils import basic_permuted_sequence_adder, negative_sequence_adder, remove_eye_sparse

from .args import Args

logger = logging.getLogger(__name__)


#


def prepare_dataset(positive_rowgen, negative_dsgen, args=None) -> DataSetGAN:
    while True:
        pos_row = next(positive_rowgen)
        pos_ds = dataset_to_gan(row_to_dataset(pos_row, 1))
        if args is not None and not dataset_matches_spec(pos_ds, args):
            continue
        ds = negative_dsgen.send(pos_ds)
        if ds is None:
            continue
        return ds


def dataset_matches_spec(ds: DataSetGAN, args: Args) -> bool:
    n_aa = ds.seqs[0].shape[1]
    if not (args.min_seq_length <= n_aa < args.max_seq_length):
        logger.debug(f"Wrong sequence length: {n_aa}.")
        return False
    adj_nodiag = remove_eye_sparse(ds.adjs[0], 3)
    frac_interactions = adj_nodiag.nnz / adj_nodiag.shape[0]
    if frac_interactions < 0.05:
        logger.debug(f"Too few interactions: {frac_interactions}.")
        return False
    return True


# Data Pipe


def check():
    indices = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    values = torch.FloatTensor([3, 4, 5])
    tensor = torch.sparse_coo_tensor(indices, values, torch.Size([2, 4]))
    return tensor


def get_data_pipe(args):
    if (
        args.training_data_cache is not None
        and args.training_data_cache.with_suffix(".index").is_file()
        and args.training_data_cache.with_suffix(".data").is_file()
    ):
        logger.info("Reading training data from cache.")
        yield from _read_ds_from_cache(args.training_data_cache)
    else:
        logger.info("Generating training data as we go.")
        yield from _generate_ds(args)


def _generate_ds(args):
    index_read, index_write = os.pipe()
    data_read, data_write = os.pipe()
    set_buf_size(data_write)
    pid = os.fork()
    if pid:
        # This is the parent process
        os.close(index_write)
        os.close(data_write)
        try:
            yield from _gen_ds_reader(index_read, data_read, args.training_data_cache)
        except Exception:
            os.close(index_read)
            os.close(data_read)
    else:
        # This is the child process
        os.close(index_read)
        os.close(data_read)
        logger.info("Before check")
        check()
        logger.info("After check")
        try:
            ds_source = get_training_datasets(args)
            _gen_ds_writer(index_write, data_write, ds_source)
        except BrokenPipeError:
            pass
        except Exception as e:
            logger.error("Caught an exception %s: %s", type(e), e)
        finally:
            os.close(index_write)
            os.close(data_write)
            # sys.stderr.flush()
            sys.stderr.close()
            sys.exit(0)


def _read_ds_from_cache(cache_file_stem: Path) -> Iterator[DataSetGAN]:
    """Iterate over datasets found in cache defined by `index_file` and `data_file`."""
    with ExitStack() as stack:
        index_fh = stack.enter_context(cache_file_stem.with_suffix(".index").open("rb"))
        data_fh = stack.enter_context(cache_file_stem.with_suffix(".data").open("rb"))
        while True:
            size_buf = index_fh.read(4)
            size = int.from_bytes(size_buf, "little")
            if size == 0:
                return
            data_buf = data_fh.read(size)
            ds = DataSetGAN.from_buffer(data_buf)
            yield ds


def _gen_ds_reader(
    index_read: int, data_read: int, cache_file_stem: Optional[Path] = None
) -> Iterator[DataSetGAN]:
    with ExitStack() as stack:
        if cache_file_stem is not None:
            logger.info("Writing generated training data to cache.")
            index_cache_fh = stack.enter_context(cache_file_stem.with_suffix(".index").open("wb"))
            data_cache_fh = stack.enter_context(cache_file_stem.with_suffix(".data").open("wb"))
        while True:
            size_buf = os.read(index_read, 4)
            size = int.from_bytes(size_buf, "little")
            if size == 0:
                return
            data_buf = os.read(data_read, size)
            if cache_file_stem is not None:
                index_cache_fh.write(size_buf)
                data_cache_fh.write(data_buf)
            ds = DataSetGAN.from_buffer(data_buf)
            yield ds


def _write_ds_to_cache(cache_file_stem: Path, ds_list: Iterator[DataSetGAN]) -> None:
    with ExitStack() as stack:
        index_fh = stack.enter_context(cache_file_stem.with_suffix(".index").open("wb"))
        data_fh = stack.enter_context(cache_file_stem.with_suffix(".data").open("wb"))
        for ds in ds_list:
            data_buf = ds.to_buffer()
            if data_buf.size == 0:
                raise Exception
            size_buf = data_buf.size.to_bytes(4, "little")
            index_fh.write(size_buf)
            data_fh.write(data_buf)


def _gen_ds_writer(index_write: int, data_write: int, ds_source: Iterator[DataSetGAN]) -> None:
    for ds in ds_source:
        data_buf = ds.to_buffer()
        size_buf = data_buf.size.to_bytes(4, "little")
        os.write(index_write, size_buf)
        os.write(data_write, data_buf)


# === Training / Validation Batches ===


def get_training_datasets(args: argparse.Namespace) -> Iterator[DataSetGAN]:
    random_state = np.random.RandomState(args.array_id)

    positive_rowgen = iter_datarows_shuffled(
        sorted(args.training_data_path.glob("database_id=*/*.parquet")),
        columns={
            "qseq": "sequence",
            "residue_idx_1_corrected": "adjacency_idx_1",
            "residue_idx_2_corrected": "adjacency_idx_2",
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
        ds = prepare_dataset(positive_rowgen, negative_dsgen, args)
        yield ds


def get_internal_validation_datasets(args: Args) -> Mapping[str, List[DataSetGAN]]:
    logger.info("Setting up validation datapipe...")
    internal_validation_datasets: Dict[str, List[DataSetGAN]] = {}
    for method in args.validation_methods.split("."):
        datagen_name = (
            f"validation_gan_{method}_{args.validation_min_seq_identity}"
            f"_{args.validation_num_sequences}"
        )
        cache_file_prefix = args.validation_cache_path.joinpath(datagen_name)
        try:
            ds_list = list(_read_ds_from_cache(cache_file_prefix))
            logger.info("Loaded validation datagen from cache: '%s'.", cache_file_prefix)
        except FileNotFoundError:
            logger.info("Generating validation datagen: '%s'.", datagen_name)
            if args.validation_data_path is None:
                raise Exception(
                    "Cached validation data not found; `validation_data_path` must be provided!"
                )
            random_state = np.random.RandomState(sum(ord(c) for c in method))
            ds_list = _get_internal_validation_dataset(args, method, random_state)
            cache_file_prefix.parent.mkdir(exist_ok=True)
            _write_ds_to_cache(cache_file_prefix, ds_list)
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
        ds = prepare_dataset(positive_rowgen, negative_dsgen, args)
        ds_list.append(ds)
    assert len(ds_list) == args.validation_num_sequences
    return ds_list
