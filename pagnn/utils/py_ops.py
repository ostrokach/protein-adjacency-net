
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def str_to_path(file: str) -> Path:
    return Path(file).resolve()


def load_yaml(file: Union[str, Path]) -> Dict[str, Any]:
    with open(file, "rt") as fin:
        data = yaml.load(fin)
    return data


def dump_yaml(data: Dict[str, Any], file: Union[str, Path]) -> Path:
    with open(file, "wt") as fout:
        yaml.dump(data, fout)
    return Path(file)
