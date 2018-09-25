import json
import logging
import pickle
import time
import gzip
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import sqlalchemy as sa
from sklearn import metrics

from pagnn.utils import StatsBase, evaluate_validation_dataset

logger = logging.getLogger(__name__)


def pickle_dump(col: Any) -> bytes:
    return pickle.dumps(col, pickle.HIGHEST_PROTOCOL)


def pickle_dump_compressed(col: Any) -> bytes:
    return gzip.compress(pickle_dump(col))


class Stats(StatsBase):
    # Class attributes
    columns_to_write: Dict[str, Optional[Callable]] = {
        "step": None,
        "stats": pickle_dump,
        "pos_preds": pickle_dump,
        "neg_preds": pickle_dump,
        "pos_losses": pickle_dump,
        "neg_losses": pickle_dump,
    }

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
        super().__init__(step)
        self.validation_time_basic = 0
        self.validation_time_extended = 0
        self._init_containers()

    def update(self) -> None:
        self.step += 1
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

    def write(self):
        # === Network parameters ===
        # for name, param in net_g.named_parameters():
        #     writer.add_histogram("net_g_" + name, to_numpy(param), self.step)
        # for name, param in net_d.named_parameters():
        #     writer.add_histogram("net_d_" + name, to_numpy(param), self.step)

        training_data = [
            ("pos_preds", self.pos_preds),
            ("neg_preds", self.neg_preds),
            ("pos_losses", self.pos_losses),
            ("neg_losses", self.neg_losses),
        ]

        # === Scalars ===
        for score_name, score_value in self.scores.items():
            if score_value is None:
                logger.warning(f"Score {score_name} is None!")
                continue
            self.writer.add_scalar(score_name, score_value, self.step)

        training_data_extended = []

        for score_name, score_value in training_data + (
            training_data_extended if self.extended else []
        ):
            self.writer.add_scalar(
                score_name + "-mean",
                np.hstack([ar.reshape(-1) for ar in score_value]).astype(np.float64).mean(),
                self.step,
            )

        # === Histograms ===
        for score_name, score_value in training_data + (
            training_data_extended if self.extended else []
        ):
            self.writer.add_histogram(
                score_name,
                np.hstack([ar.reshape(-1) for ar in score_value]).astype(np.float64),
                self.step,
            )

        # === PR Curves ===
        if self.basic:
            self.writer.add_pr_curve(
                "training_pos_pr", self.training_pos_target, self.training_pos_pred, self.step
            )
            self.writer.add_pr_curve(
                "training_fake_pr", self.training_fake_target, self.training_fake_pred, self.step
            )

        # === Text ===
        if self.extended:
            self.writer.add_text("validation_sequences", self.validation_sequences[0], self.step)
            self.writer.add_text(
                "validation_gen_sequences_0", "\n".join(self.validation_gen_sequences[0]), self.step
            )

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

        # Fake AUC
        # if self.fake_preds and self.neg_preds:
        #     self.training_fake_pred = np.hstack(
        #         [ar.mean() for ar in (self.fake_preds + self.neg_preds)]
        #     )
        #     self.training_fake_target = np.hstack(
        #         [np.ones(1) for _ in self.fake_preds] + [np.zeros(1) for _ in self.neg_preds]
        #     )
        #     self.scores["training_fake-auc"] = metrics.roc_auc_score(
        #         self.training_fake_target, self.training_fake_pred
        #     )

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
