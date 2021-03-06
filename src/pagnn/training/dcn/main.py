import itertools
import json
import logging
import os
import random
import runpy
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import sqlalchemy as sa
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim

import pagnn.models
from pagnn import init_gpu, settings
from pagnn.utils import eval_net, kill_tree

from .args import Args
from .stats import Stats
from .utils import get_data_pipe, get_internal_validation_datasets, get_training_datasets

logger = logging.getLogger(__name__)


class DatasetFinishedError(Exception):
    pass


class RuntimeExceededError(Exception):
    pass


def train(
    args: Args,
    stats: Stats,
    datapipe,
    internal_validation_datasets,
    engine: Optional[sa.engine.Engine] = None,
    checkpoint: Optional[Dict[str, Any]] = None,
    current_performance: Optional[Dict[str, Union[str, float]]] = None,
):
    """Train GAN network."""
    if checkpoint is None:
        checkpoint = {}

    # Set up network
    Net = getattr(pagnn.models.dcn, args.network_name)
    net = Net().to(settings.device)

    ds_weight = torch.tensor(
        [1.0] + [1.0 / args.num_negative_examples] * args.num_negative_examples
    )
    if args.predict_pc_identity:
        loss = nn.L1Loss(reduction="none").to(settings.device)
    else:
        loss = nn.BCELoss(reduction="none").to(settings.device)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    if args.array_id:
        net.load_state_dict(stats.load_model_state())
        write_graph = False
    elif args.model_path is not None:
        net.load_state_dict(torch.load(args.model_path))
        write_graph = False
    else:
        write_graph = True

    while True:
        current_time = time.perf_counter()

        if (current_time - stats.start_time) > args.runtime:
            raise RuntimeExceededError(
                f"Runtime exceeded! ({(current_time - stats.start_time)} > {args.runtime})"
            )
        calculate_basic_statistics = (
            stats.validation_time_basic == 0
            or (current_time - stats.validation_time_basic) > args.time_between_checkpoints
        )
        calculate_extended_statistics = calculate_basic_statistics and (
            stats.validation_time_extended == 0
            or (current_time - stats.validation_time_extended)
            > args.time_between_extended_checkpoints
        )

        # === Train discriminator ===
        net.zero_grad()

        ds_list = list(itertools.islice(datapipe, args.batch_size))
        if not ds_list:
            raise DatasetFinishedError()

        pred_list = []
        target_list = []
        error_list = []
        for ds in ds_list:
            dv = net.dataset_to_datavar(ds)
            pred = net(dv.seqs, [dv.adjs])
            pred = pred.sigmoid().mean(2).squeeze()
            error = (loss(pred, ds.targets) * ds_weight).sum()
            error.backward()
            pred_list.append(pred.detach())
            target_list.append(ds.targets.detach())
            error_list.append(error.detach())
        preds = torch.cat(pred_list)
        targets = torch.cat(target_list)
        errors = torch.stack(error_list)
        optimizer.step()

        # === Calculate Statistics ===
        if write_graph:
            # Commented out because causes errors:
            # > ** ValueError: Auto nesting doesn't know how to process an input object of type int.
            # > Accepted types: Tensors, or lists/tuples of them.
            # > ** RuntimeError: sparse tensors not supported.
            # dv = net.dataset_to_datavar(ds_list[0])
            # torch.onnx.export(
            #     net, (dv.seqs, [dv.adjs]), args.root_path.joinpath("model.onnx").as_posix()
            # )
            write_graph = False

        if calculate_basic_statistics:
            logger.debug("Calculating basic statistics...")

            stats.preds.append(preds.numpy())
            stats.targets.append(targets.numpy())
            stats.losses.append(errors.numpy())

            with torch.no_grad(), eval_net(net):
                stats.calculate_statistics_basic()

        if calculate_extended_statistics:
            logger.debug("Calculating extended statistics...")

            with torch.no_grad(), eval_net(net):
                stats.calculate_statistics_extended(net, internal_validation_datasets)

            stats.dump_model_state(net)

        if calculate_basic_statistics or calculate_extended_statistics:
            stats.write_row()
            if current_performance is not None:
                current_performance.update(stats.scores)

        stats.update()


def main(args: Optional[Args] = None):
    # === Arguments ===
    if args is None:
        args = Args.from_cli()

    logging_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}[args.verbosity]
    logging.basicConfig(format="%(message)s", level=logging_level)

    logger.info("Started training network with args: %s", args.to_dict())

    if args.custom_module:
        runpy.run_path(args.custom_module.as_posix(), globals())

    if args.gpu == -1:
        settings.device = torch.device("cpu")
        logger.info("Running on the CPU.")
    else:
        init_gpu(args.gpu)

    # === Stats ===
    db_path = args.root_path.joinpath("stats.db")
    engine = sa.create_engine(f"sqlite:///{db_path}")
    stats = Stats(engine, args)

    # === Random Seed ===
    random.seed(stats.step)
    np.random.seed(stats.step)
    torch.manual_seed(stats.step)
    torch.cuda.manual_seed(stats.step)

    # === Internal Validation Dataset ===
    logger.debug("Initializing validation dataset...")
    internal_validation_datasets = get_internal_validation_datasets(args)

    # === Training Dataset ===
    logger.debug("Initializing training dataset...")
    if True:
        datapipe = get_data_pipe(args)
    else:
        datapipe = get_training_datasets(args)

    # === Train ===
    logger.debug("Training the network...")
    start_time = time.perf_counter()
    pid = os.getpid()
    result: Dict[str, Union[str, float]] = {}

    try:
        train(args, stats, datapipe, internal_validation_datasets, current_performance=result)
    except (KeyboardInterrupt, RuntimeExceededError, DatasetFinishedError) as e:
        kill_tree(pid)
        logger.error("Training terminated with error '%s': '%s'", type(e), e)
    except Exception as e:
        kill_tree(pid)
        logger.error("Training terminated with error '%s': '%s'", type(e), e)
        raise
    finally:
        result["time_elapsed"] = time.perf_counter() - start_time
        print(json.dumps(result, sort_keys=True, indent=4))
