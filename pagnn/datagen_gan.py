# from pathlib import Path
# from typing import Generator, Iterator, List, Optional

# import numpy as np

# from .datagen import _get_rowgen_neg, _get_rowgen_pos
# from .types import DataGenGAN, DataRow, DataSetGAN


# def negative_examples_datagen(rowgen_neg: Iterator[DataRow], method: str,
#                               num_seqs: int) -> Generator[DataSet, DataSet, None]:
#     ds_with_negative = None
#     while True:
#         ds = yield ds_with_negative
#         negative_seqs = ...
#         ds_with_negative = ds._replace(seqs=ds.seqs + negative_seqs)


# def get_datagen_gan(subset: str,
#                     data_path: Path,
#                     min_seq_identity: int,
#                     methods: List[str],
#                     random_state: Optional[np.random.RandomState] = None) -> DataGen:
#     """Return a function which can generate positive or negative training examples."""
#     assert subset in ['training', 'validation', 'test']

#     datagen_pos = _get_rowgen_pos(f'adjacency_matrix_{subset}_gt{min_seq_identity}.parquet',
#                                   data_path, random_state)
#     datagen_neg = _get_rowgen_neg(
#         f'adjacency_matrix_{subset}_gt{min_seq_identity}_gbseqlen.parquet',
#         data_path, random_state)


# def add_negative_sequences(ds: DataSet,
#                            num_seqs: int,
#                            method: str,
#                            random_state: Optional[np.random.RandomState] = None) -> DataSet:
#     assert method in ['permute', 'start', 'stop', 'middle', 'edges', 'exact']


# def get_mutation_datagen(mutation_class: str, data_path: Path) -> DataGen:
#     """"""
#     ...


# def permute_and_slice_datagen(datagen_pos: Iterator[DataRow],
#                               datagen_neg: Optional[Generator[DataRow, Any, None]],
#                               methods: Tuple) -> Iterator[DataSetCollection]:
#     ...
