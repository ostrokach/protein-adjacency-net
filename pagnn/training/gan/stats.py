import time
from typing import List, Optional

import attr
import numpy as np


@attr.s
class Stats:
    validation_time = attr.ib(default=time.perf_counter())

    training_preds: List[np.ndarray] = attr.ib(default=[])
    training_targets: List[np.ndarray] = attr.ib(default=[])

    pos_losses: List[np.ndarray] = attr.ib(default=[])
    neg_losses: List[np.ndarray] = attr.ib(default=[])
    real_losses: List[np.ndarray] = attr.ib(default=[])
    fake_losses: List[np.ndarray] = attr.ib(default=[])
    g_fake_losses: List[np.ndarray] = attr.ib(default=[])

    error_g: Optional[float] = attr.ib(default=None)
    error_d: Optional[float] = attr.ib(default=None)

    scores: dict = attr.ib(default={})
