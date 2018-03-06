import functools
import logging
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, Tuple

import numpy as np

from pagnn import dataset, settings
from pagnn.exc import MaxNumberOfTriesExceededError, SequenceTooLongError
from pagnn.training.common import get_rowgen_mut, get_rowgen_neg, get_rowgen_pos
from pagnn.types import DataGen, DataRow, DataSet, DataSetCollection

logger = logging.getLogger(__name__)


def get_datagen(subset: str,
                data_path: Path,
                min_seq_identity: int,
                methods: List[str],
                random_state: Optional[np.random.RandomState] = None) -> DataGen:
    """Return a function which can generate positive or negative training examples."""
    assert subset in ['training', 'validation', 'test']

    datagen_pos = get_rowgen_pos(subset, min_seq_identity, data_path, random_state)

    if len(methods) == 1 and 'permute' in methods:
        training_datagen = functools.partial(
            permute_and_slice_datagen, datagen_pos=datagen_pos, datagen_neg=None, methods=methods)
    else:
        datagen_neg = get_rowgen_neg(subset, min_seq_identity, data_path, random_state)
        if 'permute' in methods:
            training_datagen = functools.partial(
                permute_and_slice_datagen,
                datagen_pos=datagen_pos,
                datagen_neg=datagen_neg,
                methods=methods)
        else:
            training_datagen = functools.partial(
                slice_datagen, datagen_pos=datagen_pos, datagen_neg=datagen_neg, methods=methods)

    return training_datagen


def get_mutation_datagen(mutation_class: str, data_path: Path) -> DataGen:

    mutation_datarows = get_rowgen_mut(mutation_class, data_path)
    mutation_datasets = (dataset.row_to_dataset(row, target=1) for row in mutation_datarows)

    mutation_dsc = []
    for pos_ds in mutation_datasets:
        neg_seq = bytearray(pos_ds.seq)
        mutation = pos_ds.meta['mutation']
        mutation_idx = int(mutation[1:-1]) - 1
        assert neg_seq[mutation_idx] == ord(mutation[0]), (chr(neg_seq[mutation_idx]), mutation[0])
        neg_seq[mutation_idx] = ord(mutation[-1])
        neg_ds = DataSet(neg_seq, pos_ds.adj, pos_ds.meta['score'])
        mutation_dsc.append(([pos_ds], [neg_ds]))

    def datagen():
        for dvc in mutation_dsc:
            yield dvc

    return datagen


def permute_and_slice_datagen(datagen_pos: Iterator[DataRow],
                              datagen_neg: Optional[Generator[DataRow, Any, None]],
                              methods: Tuple) -> Iterator[DataSetCollection]:
    batch_pos = []
    assert 'permute' in methods
    slice_methods = [m for m in methods if m != 'permute']
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
                            pos,
                            method=method,
                            rowgen=datagen_neg,
                        )
                        neg_list.append(other_neg)
                    except (MaxNumberOfTriesExceededError, SequenceTooLongError) as e:
                        logger.error("%s: %s", type(e), e)
                        continue
                yield pos_list, neg_list
            batch_pos = []


def slice_datagen(datagen_pos: Iterator[DataRow], datagen_neg: Generator[DataRow, Any, None],
                  methods: Tuple) -> Iterator[DataSetCollection]:
    for row in datagen_pos:
        dataset_pos = dataset.row_to_dataset(row, target=1)
        if len(dataset_pos.seq) < settings.MIN_SEQUENCE_LENGTH:
            continue
        datasets_neg = []
        try:
            for method in methods:
                dataset_neg = dataset.get_negative_example(
                    dataset_pos,
                    method=method,
                    rowgen=datagen_neg,
                )
                datasets_neg.append(dataset_neg)
        except (MaxNumberOfTriesExceededError, SequenceTooLongError) as e:
            logger.error("%s: %s", type(e), e)
            continue
        yield [dataset_pos], datasets_neg
