import gzip
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa
import torch
from sklearn import metrics

from pagnn.utils import StatsBase, evaluate_validation_dataset

from .args import Args

logger = logging.getLogger(__name__)


def pickle_dump(col: Any) -> bytes:
    return pickle.dumps(col, pickle.HIGHEST_PROTOCOL)


def pickle_dump_compressed(col: Any) -> bytes:
    return gzip.compress(pickle_dump(col))


def arrays_mean(arrays: List[np.ndarray]) -> float:
    return np.hstack([ar.reshape(-1) for ar in arrays]).astype(np.float64).mean()


def arrays_to_list(arrays: List[np.ndarray]) -> List[float]:
    return np.hstack([ar.reshape(-1) for ar in arrays]).tolist()


class Stats(StatsBase):
    step: int
    info_id: int
    batch_size: int
    root_path: Path
    start_time: float
    validation_time_basic: float
    validation_time_extended: float

    scores: Dict[str, Any]
    metadata: Dict[str, Any]

    preds: List[np.ndarray]
    targets: List[np.ndarray]
    losses: List[np.ndarray]

    basic: bool
    extended: bool

    def __init__(self, engine: sa.engine.Engine, args: Args) -> None:
        super().__init__(engine)

        self.step = self._load_step()

        self.info_id = self._write_info(self.step, args)

        self.batch_size = args.batch_size
        self.root_path = args.root_path
        self.root_path.joinpath("models").mkdir(exist_ok=True)

        self.start_time = time.perf_counter()
        self.validation_time_basic = 0
        self.validation_time_extended = 0

        self._init_containers()

    def _init_containers(self) -> None:
        # === Summary statistics ===
        self.scores = {}
        self.metadata = {}

        # === Training Data ===
        self.preds = []
        self.targets = []
        self.losses = []

        # === Basic Statistics ===
        self.basic = False
        self.training_pos_pred = None
        self.training_pos_target = None

        # === Extended Statistics ===
        self.extended = False

    def update(self) -> None:
        self.step += 1
        self._init_containers()
        logger.info("step: %s", self.step)

    def _load_step(self) -> int:
        if self._engine.has_table("stats"):
            sql_query = "SELECT max(step) step FROM stats WHERE model_location IS NOT NULL"
            step = pd.read_sql_query(sql_query, self._engine).at[0, "step"]
        else:
            step = 0
        return step

    def _write_info(self, step, args: Args) -> int:
        if self._engine.has_table("info"):
            sql_query = "select max(id) info_id from info"
            info_id = pd.read_sql_query(sql_query, self._engine).at[0, "info_id"] + 1
        else:
            info_id = 0
        info = {"id": info_id, "step": step, **args.to_dict()}
        df = pd.DataFrame(info, index=[0], columns=info.keys())
        df.to_sql("info", self._engine, if_exists="append", index=False)
        return info_id

    def get_row_data(self) -> pd.DataFrame:
        preds_array = np.hstack([ar.reshape(-1) for ar in self.preds]).astype(np.float64)
        targets_array = np.hstack([ar.reshape(-1) for ar in self.targets]).astype(np.float64)
        losses_array = np.hstack([ar.reshape(-1) for ar in self.losses]).astype(np.float64)
        data = {
            "info_id": self.info_id,
            "step": self.step,
            "sequence_number": self.step * self.batch_size,
            # Scores
            **self.scores,
            "training_pos-auc": metrics.roc_auc_score(
                np.hstack([(ar > 0.5).astype(np.int).reshape(-1) for ar in self.targets]),
                np.hstack([ar.reshape(-1) for ar in self.preds]),
            ),
            # Aggregate statistics
            "runtime": time.perf_counter() - self.start_time,
            "pos_preds-mean": preds_array[targets_array > 0.5].mean(),
            "neg_preds-mean": preds_array[targets_array <= 0.5].mean(),
            "losses-mean": losses_array.mean(),
            # Metadata (filenames, etc)
            **self.metadata,
            # TODO: Histograms
            # TODO: PR curves
            # Raw data
            "preds": pickle_dump(arrays_to_list(self.preds)),
            "targets": pickle_dump(arrays_to_list(self.targets)),
            "losses": pickle_dump(arrays_to_list(self.losses)),
        }
        return data

    def write_row(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data is None:
            data = self.get_row_data()
        df = pd.DataFrame(data, index=[0], columns=data.keys())
        df.to_sql("stats", self._engine, if_exists="append", index=False)

    def calculate_statistics_basic(self, _prev_stats={}):
        self.basic = True
        self.validation_time_basic = time.perf_counter()

        # Runtime
        prev_validation_time = _prev_stats.get("validation_time")
        _prev_stats["validation_time"] = time.perf_counter()
        if prev_validation_time:
            self.scores["time_between_checkpoints"] = time.perf_counter() - prev_validation_time
        else:
            self.scores["time_between_checkpoints"] = None

    def calculate_statistics_extended(self, net, internal_validation_datasets):
        self.extended = True
        self.validation_time_extended = time.perf_counter()

        # Validation accuracy
        for name, datasets in internal_validation_datasets.items():
            targets_valid, outputs_valid = evaluate_validation_dataset(net, datasets)
            self.scores[name + "-auc"] = metrics.roc_auc_score(targets_valid > 0.5, outputs_valid)

    def dump_model_state(self, net) -> None:
        model_location = f"models/model_{self.step:012}.state"
        model_path = self.root_path.joinpath(model_location).resolve()
        torch.save(net.state_dict(), model_path.as_posix())
        self.metadata["model_location"] = model_location

    def load_model_state(self) -> Any:
        sql_query = f"select model_location from stats where step = {self.step}"
        model_location = pd.read_sql_query(sql_query, self._engine).at[0, "model_location"]
        model_path = self.root_path.joinpath(model_location).resolve()
        return torch.load(model_path.as_posix())
