import logging

import torch
import torch.nn as nn
import yaml

from .args import Args
from .stats import Stats

logger = logging.getLogger(__name__)


def load_checkpoint(args: Args) -> dict:
    info_file = args.work_path.joinpath("info.yaml")
    checkpoint_file = args.work_path.joinpath("checkpoint.yaml")

    # Load checkpoint
    args_dict = args.to_dict()
    checkpoint: dict = {}
    if args.array_id >= 2:
        with info_file.open("rt") as fin:
            info = yaml.load(fin)
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
            checkpoint = yaml.load(fin)
        assert checkpoint["unique_name"] == args.unique_name
        assert checkpoint["step"] > 0
    else:
        if info_file.is_file():
            raise Exception(f"Info file '{info_file}' already exists!")
        with info_file.open("wt") as fout:
            yaml.dump(args_dict, fout)
    return checkpoint


def validate_checkpoint(checkpoint, scores):
    common_scores = set(checkpoint) & set(scores)
    assert common_scores
    assert all(checkpoint[s] == scores[s] for s in common_scores)


def write_checkpoint(args: Args, stats: Stats, net: nn.Module):
    if not stats.extended:
        return

    checkpoint = {"step": stats.step, "unique_name": args.unique_name, **stats.scores}

    # Save model
    net_d_dump_path = args.work_path.joinpath("models").joinpath(f"net_d-step_{stats.step}.model")
    torch.save(net.state_dict(), net_d_dump_path.as_posix())
    checkpoint["net_d_path_name"] = net_d_dump_path.name

    # Save checkpoint
    with args.work_path.joinpath("checkpoints").joinpath(f"checkpoint-step{stats.step}.json").open(
        "wt"
    ) as fout:
        yaml.dump(checkpoint, fout, sort_keys=True, indent=4)
    with args.work_path.joinpath("checkpoints").joinpath("checkpoint.json").open("wt") as fout:
        yaml.dump(checkpoint, fout, sort_keys=True, indent=4)
