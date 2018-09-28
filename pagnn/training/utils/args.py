import os
from pathlib import Path

import attr
from attr.validators import instance_of
from utils.converters import str_to_path, str_to_path_opt

from pagnn.utils import ArgsBase


class TrainingArgsBase(ArgsBase):
    # === Paths ===

    #: Location where to create subfolders for storing network data and cache files.
    root_path: Path = attr.ib(converter=str_to_path, validator=instance_of(Path))

    #: Location of the `adjacency_matrix.parquet` folder with training data.
    training_data_path: Path = attr.ib(converter=str_to_path, validator=instance_of(Path))

    #: Location of the `adjacency_matrix.parquet` folder with validation data.
    validation_data_path: Path = attr.ib(  # type: ignore
        None, converter=str_to_path_opt, validator=instance_of((Path, type(None)))  # type: ignore
    )

    validation_cache_path: Path = attr.ib(
        Path(__file__).parents[1].joinpath("data").resolve(strict=True), validator=instance_of(Path)
    )

    # === Environment ===

    array_id: int = attr.ib(int(os.getenv("SLURM_ARRAY_TASK_ID", "0")), validator=instance_of(int))
