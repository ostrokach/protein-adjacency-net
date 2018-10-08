from pathlib import Path
from typing import Optional

import attr
from attr.validators import instance_of

from pagnn import settings
from pagnn.utils import ArgsBase, load_yaml, str_to_path, str_to_path_opt


@attr.s
class PredictionArgsBase(ArgsBase):
    # === Required Arguments ===

    input_file: Path = attr.ib(converter=str_to_path)
    network_info: dict = attr.ib(converter=load_yaml)
    network_state: Path = attr.ib(converter=str_to_path)
    output_file: Optional[Path] = attr.ib(None, converter=str_to_path_opt)

    # === Optional Arguments ===

    #: Whether to show the progressbar when training.
    progressbar: bool = attr.ib(settings.SHOW_PROGRESSBAR, validator=instance_of(bool))
