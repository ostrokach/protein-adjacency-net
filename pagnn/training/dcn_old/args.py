import hashlib
import json
from pathlib import Path

import attr
from attr.validators import instance_of

import pagnn
from pagnn import settings
from pagnn.utils import ArgsBase, str_to_path


@attr.s
class Args(ArgsBase):

    # === Paths ===

    #: Location where we will create subfolders for storing network data and cache files.
    root_path: Path = attr.ib(converter=str_to_path, validator=instance_of(Path))

    #: Location of the training data.
    data_path: Path = attr.ib(converter=str_to_path, validator=instance_of(Path))

    # === Network settings ===

    #: Name of the network class.
    network_name: str = attr.ib(default="Classifier")

    #: Number of filters to include in the adjacency convolution step.
    n_filters: int = attr.ib(64, validator=instance_of(int))

    # === Optimizer settings (for Adam optimizer) ===

    #: Learning rate.
    learning_rate: float = attr.ib(0.01, validator=instance_of(float))

    #: Weight decay
    weight_decay: float = attr.ib(0.001, validator=instance_of(float))

    #: ``beta1`` for Adam.
    beta1: float = attr.ib(0.5, validator=instance_of(float))

    #: ``beta2`` for Adam.
    beta2: float = attr.ib(0.999, validator=instance_of(float))

    # === Loss settings ===

    #: Name of the loss class.
    loss_name: str = attr.ib(default="BCELoss")

    # === Dataset settings ===

    #: Number of negative sequences per batch.
    batch_size: int = attr.ib(64, validator=instance_of(int))

    #: Valid options are: ``{"permute"}``.
    training_methods: str = attr.ib("permute", validator=instance_of(str))

    #: Valid options are: ``{"seq", "adj", "seq.adj"}``.
    training_permutations: str = attr.ib(
        "seq", validator=[attr.validators.in_(["seq", "adj", "seq.adj"])]
    )

    steps_between_validation: int = attr.ib(1_000)

    # === Other things to process ===

    gpu: int = attr.ib(0, validator=instance_of(int))

    tag: str = attr.ib("", validator=instance_of(str))

    #: Array id of array jobs. 0 means that this is NOT an array job.
    array_id: int = attr.ib(0, validator=instance_of(int))

    #: Number of amino acids to process. 0 for unlimited.
    num_aa_to_process: int = attr.ib(0, validator=instance_of(int))

    #: Whether to show the progressbar when training.
    progressbar: bool = attr.ib(settings.SHOW_PROGRESSBAR, validator=instance_of(bool))

    #: Number of jobs to run concurrently (not implemented).
    num_concurrent_jobs: int = attr.ib(1, validator=instance_of(int))

    resume: bool = attr.ib(False)

    # === Cache ===

    unique_name: str = attr.ib(
        attr.Factory(lambda self: self._get_unique_name(), takes_self=True),
        validator=instance_of(str),
    )

    # === Methods ===

    @property
    def work_path(self) -> Path:
        return self.root_path.joinpath(self.unique_name)

    @property
    def network_settings(self):
        return dict(n_filters=self.n_filters)

    def _get_unique_name(self) -> str:
        state_dict = dict(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
        )
        state_bytes = json.dumps(state_dict, sort_keys=True).encode("ascii")
        # Calculating hash of dictionary: https://stackoverflow.com/a/22003440/2063031
        state_hash = hashlib.md5(state_bytes).hexdigest()[:7]
        unique_name = "-".join(
            [self.training_methods, self.training_permutations]
            + ([self.tag] if self.tag else [])
            + [pagnn.__version__, state_hash]
        )
        return unique_name
