import functools
import logging
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple

import numpy as np

from pagnn import dataset, settings
from pagnn.exc import MaxNumberOfTriesExceededError, SequenceTooLongError
from pagnn.io import iter_datarows_shuffled
from pagnn.training.common import get_rowgen
from pagnn.types import DataGen, DataRow, DataSetCollection

logger = logging.getLogger(__name__)


def get_datagen(
    data_path: Path,
    methods: List[str],
    filters: List[Tuple[Callable, str, Any]] = [],
    random_state: Optional[np.random.RandomState] = None,
) -> DataGen:
    """Return a function which can generate positive or negative training examples."""
    datagen_pos = get_rowgen(data_path, filters, False, random_state)

    if len(methods) == 1 and "permute" in methods:
        training_datagen = functools.partial(
            permute_and_slice_datagen, datagen_pos=datagen_pos, datagen_neg=None, methods=methods
        )
    else:
        datagen_neg = get_rowgen(data_path, filters, True, random_state)
        if "permute" in methods:
            training_datagen = functools.partial(
                permute_and_slice_datagen,
                datagen_pos=datagen_pos,
                datagen_neg=datagen_neg,
                methods=methods,
            )
        else:
            training_datagen = functools.partial(
                slice_datagen, datagen_pos=datagen_pos, datagen_neg=datagen_neg, methods=methods
            )

    return training_datagen


def permute_and_slice_datagen(
    datagen_pos: Iterator[DataRow],
    datagen_neg: Optional[Generator[DataRow, Any, None]],
    methods: Tuple,
) -> Iterator[DataSetCollection]:
    batch_pos = []
    assert "permute" in methods
    slice_methods = [m for m in methods if m != "permute"]
    for i, row in enumerate(datagen_pos):
        dataset_pos = dataset.row_to_dataset(row, target=1)
        if len(dataset_pos.seq) < settings.MIN_SEQUENCE_LENGTH:
            continue
        batch_pos.append(dataset_pos)
        if (i + 1) % 256 == 0:
            batch_neg = dataset.get_permuted_examples(batch_pos)
            for pos, neg in zip(batch_pos, batch_neg):
                pos_list = [pos]
                neg_list = [neg]
                for method in slice_methods:
                    try:
                        other_neg = dataset.get_negative_example(
                            pos, method=method, rowgen=datagen_neg
                        )
                        neg_list.append(other_neg)
                    except (MaxNumberOfTriesExceededError, SequenceTooLongError) as e:
                        logger.error("%s: %s", type(e), e)
                        continue
                yield pos_list, neg_list
            batch_pos = []


def slice_datagen(
    datagen_pos: Iterator[DataRow], datagen_neg: Generator[DataRow, Any, None], methods: Tuple
) -> Iterator[DataSetCollection]:
    for row in datagen_pos:
        dataset_pos = dataset.row_to_dataset(row, target=1)
        if len(dataset_pos.seq) < settings.MIN_SEQUENCE_LENGTH:
            continue
        datasets_neg = []
        try:
            for method in methods:
                dataset_neg = dataset.get_negative_example(
                    dataset_pos, method=method, rowgen=datagen_neg
                )
                datasets_neg.append(dataset_neg)
        except (MaxNumberOfTriesExceededError, SequenceTooLongError) as e:
            logger.error("%s: %s", type(e), e)
            continue
        yield [dataset_pos], datasets_neg
