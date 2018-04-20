import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import attr
from attr.validators import instance_of

import pagnn
from pagnn import settings
from pagnn.utils import ArgsBase


@attr.s
class Args(ArgsBase):

    # === Paths ===

    #: Location where to create subfolders for storing network data and cache files.
    root_path: Path = attr.ib(converter=lambda p: Path(p).resolve(), validator=instance_of(Path))

    #: Location of the `adjacency-net` databin folder.
    data_path: Path = attr.ib(
        Path(os.getenv('DATABIN_DIR')).joinpath('adjacency-net').resolve()
        if os.getenv('DATABIN_DIR') else attr.NOTHING,
        converter=lambda p: Path(p).resolve(),
        validator=instance_of(Path))

    # === Properties ===

    #: Learning rate for Discriminator (Critic).
    learning_rate_d: float = attr.ib(0.00005, validator=instance_of(float))

    #: Learning rate for Generator.
    learning_rate_g: float = attr.ib(0.00005, validator=instance_of(float))

    #: beta1 for Adam.
    beta1: float = attr.ib(0.5, validator=instance_of(float))

    #: beta2 for Adam.
    beta2: float = attr.ib(0.999, validator=instance_of(float))

    #: WGAN requires that you clamp the weights.
    clamp_lower: float = attr.ib(-0.01, validator=instance_of(float))

    #: WGAN requires that you clamp the weights.
    clamp_upper: float = attr.ib(0.01, validator=instance_of(float))

    weight_decay: float = attr.ib(0.001, validator=instance_of(float))

    # === Training Parameters ===

    hidden_size: int = attr.ib(64, validator=instance_of(int))

    #: Smallest sequence to accept for training.
    min_seq_length: int = attr.ib(64, validator=instance_of(int))

    #: Longest sequence to accept for training.
    max_seq_length: int = attr.ib(2048, validator=instance_of(int))

    #: Number of negative sequences per batch.
    batch_size: int = attr.ib(64, validator=instance_of(int))

    #: Number of steps between basic checkpoints.
    steps_between_checkpoints: int = attr.ib(5, validator=instance_of(int))

    #: Number of steps between extended checkpoints
    #: (where we evaluate performance on the validation datasets).
    steps_between_extended_checkpoints: int = attr.ib(100, validator=instance_of(int))

    #: Number of D network training iterations per round.
    d_iters: int = attr.ib(4, validator=instance_of(int))

    #: Number of G network training iterations per round.
    g_iters: int = attr.ib(1, validator=instance_of(int))

    # === Training set arguments ===

    training_methods: str = attr.ib('permute', validator=instance_of(str))
    training_min_seq_identity: int = attr.ib(0, validator=instance_of(int))
    training_permutations: str = attr.ib(
        'seq', validator=[attr.validators.in_(['seq', 'adj', 'seq.adj'])])

    # === Validation set arguments ===

    validation_methods: str = attr.ib('exact', validator=instance_of(str))
    validation_min_seq_identity: int = attr.ib(80, validator=instance_of(int))
    validation_num_sequences: int = attr.ib(1_000, validator=instance_of(int))

    # === Other things to process ===

    gpu: int = attr.ib(0, validator=instance_of(int))
    tag: str = attr.ib('', validator=instance_of(str))
    resume: bool = attr.ib(settings.ARRAY_JOB, validator=instance_of(bool))
    num_aa_to_process: Optional[int] = attr.ib(None, validator=instance_of((type(None), int)))

    #: Whether to show the progressbar when training.
    progressbar: bool = attr.ib(settings.SHOW_PROGRESSBAR, validator=instance_of(bool))

    #: Number of jobs to run concurrently (not implemented).
    num_concurrent_jobs: int = attr.ib(1, validator=instance_of(int))

    # === Cache ===

    unique_name: str = attr.ib(
        attr.Factory(lambda self: self._get_unique_name(), takes_self=True),
        validator=instance_of(str))

    # === Methods ===

    @property
    def work_path(self) -> Path:
        return self.root_path.joinpath(self.unique_name)

    def _get_unique_name(self) -> str:
        args_dict = vars(self)
        state_keys = ['learning_rate_d', 'learning_rate_g', 'weight_decay', 'hidden_size']
        # Calculating hash of dictionary: https://stackoverflow.com/a/22003440/2063031
        state_dict = {k: args_dict[k] for k in state_keys}
        state_bytes = json.dumps(state_dict, sort_keys=True).encode('ascii')
        state_hash = hashlib.md5(state_bytes).hexdigest()[:7]
        unique_name = '-'.join([
            self.training_methods,
            self.training_permutations,
            str(self.training_min_seq_identity),
        ] + ([self.tag] if self.tag else []) + [
            pagnn.__version__,
            state_hash,
        ])
        return unique_name
