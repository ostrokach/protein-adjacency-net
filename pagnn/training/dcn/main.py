import json
import logging
import random
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import sqlalchemy as sa
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import tqdm

from pagnn import init_gpu, settings
from pagnn.models import DCN
from pagnn.utils import eval_net

from .args import Args
from .stats import Stats
from .utils import generate_batch_2, get_internal_validation_datasets, get_training_datasets

logger = logging.getLogger(__name__)


class RuntimeExceededError(Exception):
    pass


def train(
    args: Args,
    stats: Stats,
    positive_rowgen,
    negative_ds_gen,
    internal_validation_datasets,
    engine: Optional[sa.engine.Engine] = None,
    checkpoint: Optional[Dict[str, Any]] = None,
    current_performance: Optional[Dict[str, Union[str, float]]] = None,
):
    """Train GAN network."""
    if checkpoint is None:
        checkpoint = {}

    # Set up network
    net = DCN("discriminator", hidden_size=args.hidden_size, bottleneck_size=1).to(settings.device)
    loss = nn.BCELoss().to(settings.device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    if args.array_id:
        net.load_state_dict(stats.load_model_state())
        write_graph = False
    else:
        write_graph = True

    progressbar = tqdm.tqdm(disable=not settings.SHOW_PROGRESSBAR)

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

        ds_list = generate_batch_2(args, net, positive_rowgen, negative_ds_gen=negative_ds_gen)

        if True:
            for ds in ds_list:
                dv = net.dataset_to_datavar(ds)
                preds = net(dv.seqs, [dv.adjs])
                preds = preds.mean(2).squeeze().sigmoid()
                targets = torch.tensor(ds.targets, dtype=torch.float)
                error = loss(preds, targets)
                error.backward()
        else:
            dv_list = [net.dataset_to_datavar(ds) for ds in ds_list]
            seqs = torch.cat([dv.seqs for dv in dv_list], 2)
            adjs = [dv.adjs for dv in dv_list]
            preds = net(seqs, adjs)
            preds = preds.mean(2).squeeze().sigmoid()
            targets = torch.tensor(ds_list[0].targets, dtype=torch.float)
            error = loss(preds, targets)
            error.backward()

        optimizer.step()

        # === Calculate Statistics ===
        if write_graph:
            # adjs_dense = [[a.to_dense() for a in adj] for adj in adjs]
            # torch.onnx.export(
            #     net, (neg_seq, adjs_dense), args.root_path.joinpath("model.onnx").as_posix()
            # )
            write_graph = False

        if calculate_basic_statistics:
            logger.debug("Calculating basic statistics...")

            stats.preds.append(preds.detach().numpy())
            stats.targets.append(targets.detach().numpy())
            stats.losses.append(error.detach().numpy())

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
        progressbar.update()


def main(args: Optional[Args] = None):
    # === Arguments ===
    if args is None:
        args = Args.from_cli()

    logging_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}[args.verbosity]
    logging.basicConfig(format="%(message)s", level=logging_level)

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
    random_state = np.random.RandomState(stats.step)

    # === Training Dataset ===
    logger.debug("Initializing training dataset...")
    positive_rowgen, negative_ds_gen = get_training_datasets(
        args, args.training_data_path, random_state
    )

    # === Internal Validation Dataset ===
    logger.debug("Initializing validation dataset...")
    internal_validation_datasets = get_internal_validation_datasets(args, args.validation_data_path)

    # === Train ===
    logger.debug("Training the network...")
    start_time = time.perf_counter()
    result: Dict[str, Union[str, float]] = {}
    try:
        train(
            args,
            stats,
            positive_rowgen,
            negative_ds_gen,
            internal_validation_datasets,
            current_performance=result,
        )
    except (KeyboardInterrupt, RuntimeExceededError) as e:
        logger.error("Training terminated with error: '%s'", e)

    result["time_elapsed"] = time.perf_counter() - start_time

    # === Output ===
    print(json.dumps(result, sort_keys=True, indent=4))
