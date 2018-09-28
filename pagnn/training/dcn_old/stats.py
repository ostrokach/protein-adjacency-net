import logging
import time
from typing import List, Optional

import numpy as np
import scipy as sp
from PIL import Image
from sklearn import metrics
from torch.autograd import Variable

from pagnn.utils import argmax_onehot, array_to_seq, score_blosum62, score_edit

from .generators import evaluate_mutation_dataset, evaluate_validation_dataset, generate_noise

logger = logging.getLogger(__name__)


class Stats:
    def __init__(self, step, writer) -> None:
        self.step = step
        self.writer = writer
        self._init_containers()
        self.validation_time_basic: float = None
        self.validation_time_extended: float = None

    def update(self) -> None:
        self.step += 1
        self._init_containers()

    def _init_containers(self) -> None:
        # === Training Data ===
        self.pos_preds: List[np.ndarray] = []
        self.neg_preds: List[np.ndarray] = []
        self.fake_preds: List[np.ndarray] = []
        self.gen_preds: List[np.ndarray] = []
        self.pos_losses: List[np.ndarray] = []
        self.neg_losses: List[np.ndarray] = []
        self.fake_losses: List[np.ndarray] = []
        self.gen_losses: List[np.ndarray] = []

        self.scores: dict = {}

        # === Basic Statistics ===
        self.basic: bool = False
        self.training_pos_pred: Optional[np.ndarray] = None
        self.training_pos_target: Optional[np.ndarray] = None
        self.training_fake_pred: Optional[np.ndarray] = None
        self.training_fake_target: Optional[np.ndarray] = None

        # === Extended Statistics ===
        self.extended: bool = False
        self.blosum62_scores: List[np.ndarray] = []
        self.edit_scores: List[np.ndarray] = []
        self.validation_sequences: List[str] = []
        self.validation_gen_sequences: List[List[str]] = []

    def write(self):
        # === Network parameters ===
        # for name, param in net_g.named_parameters():
        #     writer.add_histogram("net_g_" + name, param.numpy(), self.step)
        # for name, param in net_d.named_parameters():
        #     writer.add_histogram("net_d_" + name, param.numpy(), self.step)

        training_data = [
            ("pos_preds", self.pos_preds),
            ("neg_preds", self.neg_preds),
            ("fake_preds", self.fake_preds),
            ("gen_preds", self.gen_preds),
            ("pos_losses", self.pos_losses),
            ("neg_losses", self.neg_losses),
            ("fake_losses", self.fake_losses),
            ("gen_losses", self.gen_losses),
        ]

        training_data_extended = [
            ("blosum62_scores", self.blosum62_scores),
            ("edit_scores", self.edit_scores),
        ]

        # === Scalars ===
        for score_name, score_value in self.scores.items():
            if score_value is None:
                logger.warning(f"Score {score_name} is None!")
                continue
            self.writer.add_scalar(score_name, score_value, self.step)

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
        if self.fake_preds and self.neg_preds:
            self.training_fake_pred = np.hstack(
                [ar.mean() for ar in (self.fake_preds + self.neg_preds)]
            )
            self.training_fake_target = np.hstack(
                [np.ones(1) for _ in self.fake_preds] + [np.zeros(1) for _ in self.neg_preds]
            )
            self.scores["training_fake-auc"] = metrics.roc_auc_score(
                self.training_fake_target, self.training_fake_pred
            )

        # Runtime
        prev_validation_time = _prev_stats.get("validation_time")
        _prev_stats["validation_time"] = time.perf_counter()
        if prev_validation_time:
            self.scores["time_between_checkpoints"] = time.perf_counter() - prev_validation_time

    def calculate_statistics_extended(
        self, net_d, net_g, internal_validation_datasets, external_validation_datasets
    ):
        self.extended = True
        self.validation_time_extended = time.perf_counter()

        # Validation accuracy
        for name, datasets in internal_validation_datasets.items():
            targets_valid, outputs_valid = evaluate_validation_dataset(net_d, datasets)
            self.scores[name + "-auc"] = metrics.roc_auc_score(targets_valid, outputs_valid)

        for name, datasets in external_validation_datasets.items():
            targets_valid, outputs_valid = evaluate_mutation_dataset(net_d, datasets)
            if "protherm" in name:
                # Protherm predicts ΔΔG, so positive values are destabilizing
                self.scores[name + "-spearman_corr"] = sp.stats.spearmanr(
                    -targets_valid, outputs_valid
                ).correlation
            elif "humsavar" in name:
                # For humsavar: 0 = stable, 1 = deleterious
                self.scores[name + "-auc"] = metrics.roc_auc_score(1 - targets_valid, outputs_valid)
            else:
                self.scores[name + "-auc"] = metrics.roc_auc_score(targets_valid + 1, outputs_valid)

        for name in ["validation_gan_exact_80_1000"]:
            for i, dataset in enumerate(internal_validation_datasets[name]):
                seq_wt = dataset.seqs[0].decode()
                start = 0
                stop = start + len(seq_wt)
                datavar = net_g.dataset_to_datavar(dataset)
                noise = generate_noise(net_g, [datavar.adjs])
                noisev = Variable(noise.normal_(0, 1))
                pred = net_g(noisev, [datavar.adjs])
                pred_argmax = argmax_onehot(pred[:, :, start:stop].data)
                target = datavar.seqs[0, :, start:stop].data

                self.blosum62_scores.append(score_blosum62(target, pred_argmax))
                self.edit_scores.append(score_edit(target, pred_argmax))
                self.validation_sequences.append(seq_wt)
                self.validation_gen_sequences.append(
                    [array_to_seq(pred[i, :, start:stop].numpy()) for i in range(pred.shape[0])]
                )
