import argparse
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import attr

T = TypeVar('T', bound='ArgsBase')


@attr.s
class ArgsBase:

    @classmethod
    def from_cli(cls: Type[T]) -> T:
        parser = argparse.ArgumentParser()
        for attribute in cls.__attrs_attrs__:  # type: ignore
            title = f"--{attribute.name.replace('_', '-')}"
            kwargs: Dict[str, Any] = {}
            if attribute.default is attr.NOTHING:
                kwargs['required'] = True
            elif isinstance(attribute.default, attr.Factory):
                kwargs['default'] = attr.NOTHING
            else:
                kwargs['default'] = attribute.default
            if attribute.type is bool:
                if attribute.default is False:
                    kwargs['action'] = 'store_true'
                elif attribute.default is True:
                    kwargs['action'] = 'store_false'
                else:
                    raise Exception("A boolean attribute must have a default value that is either "
                                    "`True` or `False`.")
            if 'action' not in kwargs:
                kwargs['type'] = attribute.type
            parser.add_argument(title, **kwargs)
        args = parser.parse_args()
        return cls(**vars(args))  # type: ignore

    @classmethod
    def from_json(cls: Type[T], data: dict) -> T:
        args = dict()
        for attribute in cls.__attrs_attrs__:  # type: ignore
            args[attribute.name] = data[attribute.name]
        return cls(**args)  # type: ignore

    def to_json(self) -> dict:
        data = dict()
        for attribute in self.__attrs_attrs__:  # type: ignore
            if attribute.type is Path:
                value = getattr(self, attribute.name).as_posix()
            else:
                value = getattr(self, attribute.name)
            data[attribute.name] = value
        return data
