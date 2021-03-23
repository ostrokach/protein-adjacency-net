from pathlib import Path
from typing import Optional

import attr
from attr.validators import instance_of

from pagnn import settings
from pagnn.utils import ArgsBase, load_yaml, str_to_path, str_to_path_opt


@attr.s(kw_only=True)
class PredictionArgsBase(ArgsBase):
    # === Required Arguments ===

    network_info: dict = attr.ib(converter=load_yaml, validator=instance_of(dict))
    network_state: Path = attr.ib(converter=str_to_path, validator=instance_of(Path))
    input_file: Optional[Path] = attr.ib(
        None, converter=str_to_path_opt, validator=instance_of((type(None), Path))
    )
    output_file: Optional[Path] = attr.ib(
        None, converter=str_to_path_opt, validator=instance_of((type(None), Path))
    )

    # === Optional Arguments ===

    #: Whether to show the progressbar when training.
    progressbar: bool = attr.ib(settings.SHOW_PROGRESSBAR, validator=instance_of(bool))
