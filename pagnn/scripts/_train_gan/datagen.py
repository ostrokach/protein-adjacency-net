# from typing import Iterator, Generator

# from pagnn.types import DataRow, DataSetGAN


# def add_permuted_sequence(rowgen: Iterator[DataRow], num_sequences: int,
#                           random_state) -> Generator[DataSetGAN, DataSetGAN, None]:
#     """

#     Args:
#         rowgen: Used for **pre-populating** the generator only!
#         num_sequences: Number of sequences to generate in each iteration.
#     """
#     random_state = ...


# def _add_permuted_sequence():
#     ...


# def add_extracted_sequence(rowgen: Iterator[DataRow], method: str, num_sequences: int,
#                            random_state) -> Generator[DataSetGAN, DataSetGAN, None]:
#     """

#     Args:
#         rowgen: Generator used for fetching negative rows.
#         method: Method by which longer negative sequences get cut to the correct size.
#         num_sequences: Number of sequences to generate in each iteration.
#     """
#     random_state = ...


# def _add_negative_sequence():
#     negative_ds = get_negative_example(DataSet(
#         dsg.seqs[0],
#         dsg.adj,
#     ))


# def add_generated_sequence(net: Generator, num_sequences):
#     ...
