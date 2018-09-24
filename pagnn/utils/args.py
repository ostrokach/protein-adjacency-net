import argparse
import os
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import attr
from attr.validators import instance_of

from .converters import str_to_path, str_to_path_opt

T = TypeVar("T", bound="ArgsBase")


@attr.s
class ArgsBase:

    # === Paths ===

    #: Location where to create subfolders for storing network data and cache files.
    root_path: Path = attr.ib(converter=str_to_path, validator=instance_of(Path))

    #: Location of the `adjacency_matrix.parquet` folder with training data.
    training_data_path: Path = attr.ib(converter=str_to_path, validator=instance_of(Path))

    #: Location of the `adjacency_matrix.parquet` folder with validation data.
    validation_data_path: Path = attr.ib(  # type: ignore
        None, converter=str_to_path_opt, validator=instance_of((Path, type(None)))  # type: ignore
    )

    # === Environment ===

    array_id: int = attr.ib(int(os.getenv("SLURM_ARRAY_TASK_ID", "0")), validator=instance_of(int))

    # === Serialization ===

    @classmethod
    def from_cli(cls: Type[T]) -> T:
        parser = argparse.ArgumentParser()
        for attribute in cls.__attrs_attrs__:  # type: ignore
            title = f"--{attribute.name.replace('_', '-')}"
            kwargs: Dict[str, Any] = {}
            if attribute.default is attr.NOTHING:
                kwargs["required"] = True
            elif isinstance(attribute.default, attr.Factory):  # type: ignore
                kwargs["default"] = attr.NOTHING
            else:
                kwargs["default"] = attribute.default
            if attribute.type is bool:
                if attribute.default is False:
                    kwargs["action"] = "store_true"
                elif attribute.default is True:
                    kwargs["action"] = "store_false"
                else:
                    raise Exception(
                        "A boolean attribute must have a default value that is either "
                        "`True` or `False`."
                    )
            if attribute.type is not bool and attribute.converter is None:
                kwargs["type"] = attribute.type
            parser.add_argument(title, **kwargs)
        args = parser.parse_args()
        return cls(**vars(args))  # type: ignore

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        args = dict()
        for attribute in cls.__attrs_attrs__:  # type: ignore
            args[attribute.name] = data[attribute.name]
        return cls(**args)  # type: ignore

    def to_dict(self) -> dict:
        data = dict()
        for attribute in self.__attrs_attrs__:  # type: ignore
            if attribute.type is Path:
                value = getattr(self, attribute.name).as_posix()
            else:
                value = getattr(self, attribute.name)
            data[attribute.name] = value
        return data
