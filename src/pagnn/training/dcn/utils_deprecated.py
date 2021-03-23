import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator, Optional

import torch

from pagnn.datapipe import set_buf_size
from pagnn.training.dcn.utils import _read_ds_from_cache, get_training_datasets
from pagnn.types import DataSetGAN

logger = logging.getLogger(__name__)


def check_for_deadlock():
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
    """Old version used with os.Fork."""
    index_read, index_write = os.pipe()
    data_read, data_write = os.pipe()
    set_buf_size(data_write)
    pid = os.fork()
    if pid:
        # This is the parent process
        os.close(index_write)
        os.close(data_write)
        try:
            yield from _gen_ds_reader(index_read, data_read)
        except Exception:
            os.close(index_read)
            os.close(data_read)
    else:
        # This is the child process
        os.close(index_read)
        os.close(data_read)
        logger.info("Before checking for deadlock.")
        check_for_deadlock()
        logger.info("After checking for deadlock.")
        try:
            ds_source = get_training_datasets(args)
            _gen_ds_writer(index_write, data_write, ds_source, args.training_data_cache)
        except BrokenPipeError:
            # The receiving process terminated
            pass
        except Exception as e:
            logger.error("Caught an exception %s: %s", type(e), e)
        finally:
            os.close(index_write)
            os.close(data_write)
            sys.stderr.close()
            sys.exit(0)


def _gen_ds_reader(index_read: int, data_read: int) -> Iterator[DataSetGAN]:
    while True:
        size_buf = os.read(index_read, 4)
        size = int.from_bytes(size_buf, "little")
        if size == 0:
            return
        data_buf = os.read(data_read, size)
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


def _gen_ds_writer(
    index_write: int,
    data_write: int,
    ds_source: Iterator[DataSetGAN],
    cache_file_stem: Optional[Path] = None,
) -> None:
    with ExitStack() as stack:
        if cache_file_stem is not None:
            logger.info("Writing generated training data to cache.")
            index_cache_fh = stack.enter_context(cache_file_stem.with_suffix(".index").open("wb"))
            data_cache_fh = stack.enter_context(cache_file_stem.with_suffix(".data").open("wb"))
        for ds in ds_source:
            data_buf = ds.to_buffer()
            size_buf = data_buf.size.to_bytes(4, "little")
            os.write(index_write, size_buf)
            os.write(data_write, data_buf)
            if cache_file_stem is not None:
                index_cache_fh.write(size_buf)
                data_cache_fh.write(data_buf)
