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
from pagnn.utils import eval_net, load_checkpoint, to_numpy, validate_checkpoint, write_checkpoint

from .args import Args
from .stats import Stats
from .utils import generate_batch, get_internal_validation_datasets, get_training_datasets

logger = logging.getLogger(__name__)


def train(
    args: Args,
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

    args.root_path.joinpath("models").mkdir(exist_ok=True)
    args.root_path.joinpath("checkpoints").mkdir(exist_ok=True)

    # Set up network
    net_d = DCN("discriminator", hidden_size=args.hidden_size, bottleneck_size=1)
    net_d = net_d.to(settings.device)

    loss = nn.BCELoss().to(settings.device)
    # one = torch.tensor(1, dtype=torch.float, device=settings.device)
    # mone = torch.tensor(-1, dtype=torch.float, device=settings.device)

    optimizer_d = optim.Adam(
        net_d.parameters(), lr=args.learning_rate_d, betas=(args.beta1, args.beta2)
    )

    if args.array_id:
        net_d.load_state_dict(
            torch.load(
                args.root_path.joinpath("models").joinpath(checkpoint["net_d_path_name"]).as_posix()
            )
        )
        write_graph = False
    else:
        write_graph = True

    step = checkpoint.get("step", 0)
    stats = Stats(step, engine)
    progressbar = tqdm.tqdm(disable=not settings.SHOW_PROGRESSBAR)

    while True:
        # num_seqs_processed = (stats.step * (args.d_iters * 3 + args.g_iters) * args.batch_size)

        calculate_basic_statistics = (
            stats.validation_time_basic == 0
            or (time.perf_counter() - stats.validation_time_basic) > args.time_between_checkpoints
        )
        calculate_extended_statistics = calculate_basic_statistics and (
            stats.validation_time_extended == 0
            or (time.perf_counter() - stats.validation_time_extended)
            > args.time_between_extended_checkpoints
        )

        # === Train discriminator ===
        logger.debug("Zeroing out the network...")
        net_d.zero_grad()

        logger.debug("Generating batch...")
        pos_seq, neg_seq, adjs = generate_batch(
            args, net_d, positive_rowgen, negative_ds_gen=negative_ds_gen
        )

        # adjs = [[a.to_dense() for a in adj] for adj in adjs]

        # Pos
        logger.debug("Training on +ive examples...")
        pos_pred = net_d(pos_seq, adjs).sigmoid()
        pos_target = torch.ones(pos_pred.shape, device=settings.device)
        pos_loss = loss(pos_pred, pos_target)
        pos_loss.backward()

        # Neg
        logger.debug("Training on -ive examples...")
        neg_pred = net_d(neg_seq, adjs).sigmoid()
        neg_target = torch.zeros(neg_pred.shape, device=settings.device)
        neg_loss = loss(neg_pred, neg_target)
        neg_loss.backward()

        optimizer_d.step()

        # === Calculate Statistics ===
        if calculate_basic_statistics:
            stats.pos_preds.append(to_numpy(pos_pred))
            stats.neg_preds.append(to_numpy(neg_pred))
            stats.pos_losses.append(to_numpy(pos_loss))
            stats.neg_losses.append(to_numpy(neg_loss))

            if write_graph:
                # TODO: Uncomment when this works in tensorboardX
                # torch.onnx.export(net_d, (pos_seq, adjs), "alexnet.onnx", verbose=True)
                # writer.add_graph(net_d, (pos_seq, adjs, ), verbose=True)
                # writer.add_graph(net_g, (noisev, adjs))
                write_graph = False

        # === Write Statistics ===
        if calculate_basic_statistics:
            resume_checkpoint = args.array_id and step == checkpoint.get("step")

            # Basic statistics
            with torch.no_grad(), eval_net(net_d):
                stats.calculate_statistics_basic()

                if calculate_extended_statistics:
                    stats.calculate_statistics_extended(net_d, internal_validation_datasets)

            # Write to disk
            if resume_checkpoint:
                validate_checkpoint(checkpoint, stats.scores)
            else:
                stats.write()
                write_checkpoint(args, stats, net_d)
                if current_performance is not None:
                    current_performance.update(writer.scalar_dict)

        stats.update()
        progressbar.update()


def main():
    # === Arguments ===
    args = Args.from_cli()

    logging_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}[args.verbosity]
    logging.basicConfig(format="%(message)s", level=logging_level)

    if args.gpu == -1:
        settings.device = torch.device("cpu")
        logger.info("Running on the CPU.")
    else:
        init_gpu(args.gpu)

    # === Random Seed ===
    random.seed(42 + args.array_id)
    np.random.seed(42 + args.array_id)
    torch.manual_seed(42 + args.array_id)
    torch.cuda.manual_seed(42 + args.array_id)
    random_state = np.random.RandomState(42)

    # === Ststs ===
    db_path = args.root_path.joinpath("stats.db")
    engine = sa.create_engine(f"sqlite:///{db_path}")

    # === Checkpoint ===
    checkpoint = load_checkpoint(args)

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
            positive_rowgen,
            negative_ds_gen,
            internal_validation_datasets,
            engine,
            checkpoint,
            current_performance=result,
        )
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
    result["time_elapsed"] = time.perf_counter() - start_time

    # === Output ===
    print(json.dumps(result, sort_keys=True, indent=4))
