import os
from pathlib import Path
from typing import Optional

import attr
from attr.validators import instance_of

from pagnn import settings
from pagnn.training.utils import TrainingArgsBase
from pagnn.utils import str_to_path_opt, str_to_seconds


@attr.s
class Args(TrainingArgsBase):

    # === Parameters to Tune ===
    network_name: str = attr.ib("DCN", validator=instance_of(str))
    num_negative_examples: int = attr.ib(63, validator=instance_of(int))
    n_layers: int = attr.ib(3, validator=instance_of(int))

    custom_module: Optional[Path] = attr.ib(
        None, converter=str_to_path_opt, validator=instance_of((type(None), Path))
    )

    #: Specify the model to start with (useful for debugging only!)
    model_path: Optional[Path] = attr.ib(
        None, converter=str_to_path_opt, validator=instance_of((type(None), Path))
    )

    # === Properties ===

    #: Learning rate for Classifier.
    learning_rate: float = attr.ib(0.00005, validator=instance_of(float))

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

    #: Number of seconds between basic checkpoints (default = ``1m``).
    time_between_checkpoints: float = attr.ib(  # type: ignore
        "1m", converter=str_to_seconds, validator=instance_of(float)  # type: ignore
    )

    #: Number of seconds between extended checkpoints (default = ``10m``).
    #: (where we evaluate performance on the validation datasets).
    time_between_extended_checkpoints: float = attr.ib(  # type: ignore
        "10m", converter=str_to_seconds, validator=instance_of(float)  # type: ignore
    )

    #: Number of seconds after which training should be terminated
    #: (default = ``999d | ${SBATCH_TIMELIMIT} - 20min``).
    #: Note that we subtract ``20min`` to give the job some time for cleanup.
    runtime: float = attr.ib(
        max(600.0, str_to_seconds(os.getenv("SBATCH_TIMELIMIT", "999d")) - 1200.0),
        converter=str_to_seconds,
        validator=instance_of(float),  # type: ignore
    )

    #: Number of D network training iterations per round.
    d_iters: int = attr.ib(1, validator=instance_of(int))

    #: Number of G network training iterations per round.
    g_iters: int = attr.ib(1, validator=instance_of(int))

    # === Training set arguments ===

    #: Fraction of available training data (measured as the number of Parquet files)
    #: to use for training the network.
    subsample_training_data: float = attr.ib(1, validator=instance_of(float))

    #: Permute the sequence and adjacency matrix of positive training examples.
    permute_positives: bool = attr.ib(False, validator=instance_of(bool))

    #: Predict (adjusted) % identity of a multiple sequence alignment,
    #: rather than 1 or 0 when the sequence matches or does not match the adjacency.
    predict_pc_identity: bool = attr.ib(False, validator=instance_of(bool))

    training_methods: str = attr.ib("permute", validator=instance_of(str))
    training_min_seq_identity: int = attr.ib(0, validator=instance_of(int))
    training_permutations: str = attr.ib(
        "seq", validator=[attr.validators.in_(["seq", "adj", "seq.adj"])]
    )

    # === Validation set arguments ===

    validation_methods: str = attr.ib("permute.exact", validator=instance_of(str))
    validation_min_seq_identity: int = attr.ib(80, validator=instance_of(int))
    validation_num_sequences: int = attr.ib(1000, validator=instance_of(int))

    # === Other things to process ===

    gpu: int = attr.ib(0, validator=instance_of(int))
    tag: str = attr.ib("", validator=instance_of(str))

    #: Array id of array jobs. 0 means that this is NOT an array job.
    array_id: int = attr.ib(int(os.getenv("SLURM_ARRAY_TASK_ID", "0")), validator=instance_of(int))

    num_sequences_to_process: int = attr.ib(0, validator=instance_of(int))

    #: Whether to show the progressbar when training.
    progressbar: bool = attr.ib(settings.SHOW_PROGRESSBAR, validator=instance_of(bool))

    #: Number of jobs to run concurrently (not implemented).
    num_concurrent_jobs: int = attr.ib(1, validator=instance_of(int))

    #:
    verbosity: int = attr.ib(1, validator=instance_of(int))
