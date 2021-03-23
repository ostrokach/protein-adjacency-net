import json
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .args import ArgsBase
from .stats import StatsBase

logger = logging.getLogger(__name__)


def load_checkpoint(args: ArgsBase) -> dict:
    info_file = args.root_path.joinpath("info.json")
    checkpoint_file = args.root_path.joinpath("checkpoint.json")

    # Load checkpoint
    args_dict = args.to_dict()
    checkpoint: dict = {}
    if args.array_id >= 2:
        with info_file.open("rt") as fin:
            info = json.load(fin)
        assert info["array_id"] == args.array_id - 1
        for key in info:
            if key in ["array_id"]:
                continue
            if info[key] != args_dict.get(key):
                logger.warning(
                    "The value for parameter '%s' is different from the previous run. "
                    "('%s' != '%s')",
                    key,
                    info[key],
                    args_dict.get(key),
                )
        with checkpoint_file.open("rt") as fin:
            checkpoint = json.load(fin)
        assert checkpoint["step"] > 0
    else:
        if info_file.is_file():
            logger.warning(f"Info file '{info_file}' already exists!")
        with info_file.open("wt") as fout:
            json.dump(args_dict, fout, sort_keys=True, indent=4)
    return checkpoint


def validate_checkpoint(checkpoint, scores):
    common_scores = set(checkpoint) & set(scores)
    assert common_scores
    assert all(checkpoint[s] == scores[s] for s in common_scores)


def write_checkpoint(
    args: ArgsBase, stats: StatsBase, net_d: nn.Module, net_g: Optional[nn.Module] = None
):
    if not stats.extended:
        return

    checkpoint: Dict[str, Any] = {"step": stats.step, **stats.scores}

    # Save model
    net_d_dump_path = args.root_path.joinpath("models").joinpath(f"net_d-step_{stats.step}.model")
    torch.save(net_d.state_dict(), net_d_dump_path.as_posix())
    checkpoint["net_d_path_name"] = net_d_dump_path.name

    if net_g is not None:
        net_g_dump_path = args.root_path.joinpath("models").joinpath(
            f"net_g-step_{stats.step}.model"
        )
        torch.save(net_g.state_dict(), net_g_dump_path.as_posix())
        checkpoint["net_g_path_name"] = net_g_dump_path.name

    # Save checkpoint
    with args.root_path.joinpath("checkpoints").joinpath(f"checkpoint-step{stats.step}.json").open(
        "wt"
    ) as fout:
        json.dump(checkpoint, fout, sort_keys=True, indent=4)
    with args.root_path.joinpath("checkpoints").joinpath("checkpoint.json").open("wt") as fout:
        json.dump(checkpoint, fout, sort_keys=True, indent=4)
