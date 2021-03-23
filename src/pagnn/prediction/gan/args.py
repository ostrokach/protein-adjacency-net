from pathlib import Path

import attr
from attr.validators import instance_of

from pagnn.utils import ArgsBase


def str_to_path(file):
    return Path(file).resolve()


@attr.s
class Args(ArgsBase):

    #: File from which we should read input (either *.pdb or *.parquet).
    input_file: Path = attr.ib(
        converter=lambda p: Path(p).resolve(), validator=instance_of(Path)
    )

    #: File to which we should save the output.
    output_file: Path = attr.ib(
        converter=lambda p: Path(p), validator=instance_of(Path)
    )

    #: Work path that was used for training (``args_training.work_path``).
    work_path: Path = attr.ib(
        converter=lambda p: Path(p).resolve(), validator=instance_of(Path)
    )

    #: Step for which to import the model.
    step: int = attr.ib(validator=instance_of(int))

    #: Number of sequences to generate.
    nseqs: int = attr.ib(validator=instance_of(int))

    #: Max number of cores to use.
    nprocs: int = attr.ib(None, validator=instance_of((type(None), int)))

    #: Whether to predict secondary structure for generated sequences.
    include_ss: bool = attr.ib(False, validator=instance_of(bool))

    #: Number of sequences per batch.
    batch_size: int = attr.ib(256, validator=instance_of(int))

    # === ===
    #:
    validation_dataset_file: Path = attr.ib(
        "", converter=lambda p: Path(p).resolve(), validator=instance_of(Path)
    )
