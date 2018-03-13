import time
from typing import List, Optional

import attr
import numpy as np
from PIL import Image


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

    scores: dict = attr.ib(default={})

    #: Flag documenting whether we include extended statistics
    is_extended: bool = attr.ib(default=False)

    blosum62_scores: List[float] = attr.ib(default=[])
    edit_scores: List[float] = attr.ib(default=[])

    weblogo1: Optional[Image.Image] = attr.ib(default=None)
