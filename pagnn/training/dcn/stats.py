import gzip
import logging
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn import metrics

from pagnn.utils import StatsBase, evaluate_validation_dataset

logger = logging.getLogger(__name__)


def pickle_dump(col: Any) -> bytes:
    return pickle.dumps(col, pickle.HIGHEST_PROTOCOL)


def pickle_dump_compressed(col: Any) -> bytes:
    return gzip.compress(pickle_dump(col))


def arrays_mean(arrays: List[np.ndarray]) -> float:
    return np.hstack([ar.reshape(-1) for ar in score_value]).astype(np.float64).mean()


class Stats(StatsBase):
    # Instance attributes
    step: int
    engine: sa.engine.Engine

    validation_time_basic: float
    validation_time_extended: float

    scores: Dict[str, Any]
    pos_preds: List[np.ndarray]
    neg_preds: List[np.ndarray]
    pos_losses: List[np.ndarray]
    neg_losses: List[np.ndarray]
    basic: bool
    training_pos_pred: Optional[np.ndarray]
    training_pos_target: Optional[np.ndarray]
    extended: bool

    def __init__(self, step: int, engine: sa.engine.Engine) -> None:
        super().__init__(step, engine)
        self.validation_time_basic = 0
        self.validation_time_extended = 0
        self._init_containers()

    def _init_containers(self) -> None:
        # === Summary statistics ===
        self.scores = {}

        # === Training Data ===
        self.pos_preds = []
        self.neg_preds = []
        self.pos_losses = []
        self.neg_losses = []

        # === Basic Statistics ===
        self.basic = False
        self.training_pos_pred = None
        self.training_pos_target = None

        # === Extended Statistics ===
        self.extended = False

    def update(self) -> None:
        self.step += 1
        self._init_containers()

    def get_row(self) -> pd.DataFrame:
        data = {
            "step": self.step,
            # Scores
            **self.scores,
            # Aggregate statistics
            "pos_preds-mean": arrays_mean(self.pos_preds),
            "neg_preds-mean": arrays_mean(self.neg_preds),
            "pos_losses-mean": arrays_mean(self.pos_losses),
            "neg_losses-mean": arrays_mean(self.neg_losses),
            # TODO: Histograms
            # TODO: PR curves
            # Raw data
            "pos_preds": pickle_dump(self.pos_preds),
            "neg_preds": pickle_dump(self.neg_preds),
            "pos_losses": pickle_dump(self.pos_losses),
            "neg_losses": pickle_dump(self.neg_losses),
        }
        df = pd.DataFrame(data, index=[0], columns=list(data.values()))
        return df

    def write_row(self) -> None:
        df = self.get_row()
        df.to_sql("stats", self._engine, if_exists="append", index=False)

    def calculate_statistics_basic(self, _prev_stats={}):
        self.basic = True
        self.validation_time_basic = time.perf_counter()

        # Pos AUC
        if self.pos_preds and self.neg_preds:
            self.training_pos_pred = np.hstack(
                [ar.mean() for ar in (self.pos_preds + self.neg_preds)]
            )
            self.training_pos_target = np.hstack(
                [np.ones(1) for _ in self.pos_preds] + [np.zeros(1) for _ in self.neg_preds]
            )
            self.scores["training_pos-auc"] = metrics.roc_auc_score(
                self.training_pos_target, self.training_pos_pred
            )

        # Runtime
        prev_validation_time = _prev_stats.get("validation_time")
        _prev_stats["validation_time"] = time.perf_counter()
        if prev_validation_time:
            self.scores["time_between_checkpoints"] = time.perf_counter() - prev_validation_time

    def calculate_statistics_extended(self, net_d, internal_validation_datasets):
        self.extended = True
        self.validation_time_extended = time.perf_counter()

        # Validation accuracy
        for name, datasets in internal_validation_datasets.items():
            targets_valid, outputs_valid = evaluate_validation_dataset(net_d, datasets)
            self.scores[name + "-auc"] = metrics.roc_auc_score(targets_valid, outputs_valid)
