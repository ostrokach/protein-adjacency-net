import json
import logging
import random
import time
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from pagnn import init_gpu, settings
from pagnn.models import AESeqAdjApplyExtra
from pagnn.utils import (
    eval_net,
    freeze_net,
    load_checkpoint,
    to_numpy,
    unfreeze_net,
    validate_checkpoint,
    write_checkpoint,
)

from .args import Args
from .stats import Stats
from .utils import (
    generate_batch,
    generate_noise,
    get_internal_validation_datasets,
    get_training_datasets,
)

logger = logging.getLogger(__name__)


def train(
    args: Args,
    writer: SummaryWriter,
    positive_rowgen,
    negative_ds_gen,
    internal_validation_datasets,
    checkpoint,
    current_performance: Optional[Dict[str, Union[str, float]]] = None,
):
    """Train GAN network."""
    args.root_path.joinpath("models").mkdir(exist_ok=True)
    args.root_path.joinpath("checkpoints").mkdir(exist_ok=True)

    # Set up network
    net = AESeqAdjApplyExtra("discriminator", hidden_size=args.hidden_size, bottleneck_size=1).to(
        settings.device
    )

    loss = nn.BCELoss().to(settings.device)
    one = torch.ones(1, device=settings.device)
    mone = one * -1

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    if args.array_id:
        net.load_state_dict(
            torch.load(
                args.root_path.joinpath("models").joinpath(checkpoint["net_path_name"]).as_posix()
            )
        )
        write_graph = False
    else:
        write_graph = True

    step = checkpoint.get("step", 0)
    stats = Stats(step, writer)
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
        net.zero_grad()

        pos_seq, neg_seq, adjs = generate_batch(
            args, net, positive_rowgen, negative_ds_gen=negative_ds_gen
        )

        # Pos
        pos_pred = net(pos_seq, adjs).sigmoid()
        pos_target = torch.ones(pos_pred.shape).to(settings.device)
        pos_loss = loss(pos_pred, pos_target)
        pos_loss.backward(one * 2)

        # Neg
        neg_pred = net(neg_seq, adjs).sigmoid()
        neg_target = torch.zeros(neg_pred.shape).to(settings.device)
        neg_loss = loss(neg_pred, neg_target)
        neg_loss.backward(one)

        # Fake
        noise = generate_noise(net_g, adjs)
        noisev = Variable(noise.normal_(0, 1))
        with torch.no_grad():
            fake_seq = net_g(noisev, adjs).data
        fake_seq = Variable(fake_seq)
        fake_pred = net_d(fake_seq, adjs).sigmoid()
        fake_target = torch.zeros(fake_pred.shape).to(settings.device)
        fake_loss = loss(fake_pred, fake_target)
        fake_loss.backward(one)

        optimizer_d.step()

        # Update stats
        if calculate_basic_statistics:
            stats.pos_preds.append(to_numpy(pos_pred))
            stats.neg_preds.append(to_numpy(neg_pred))
            stats.fake_preds.append(to_numpy(fake_pred))
            stats.pos_losses.append(to_numpy(pos_loss))
            stats.neg_losses.append(to_numpy(neg_loss))
            stats.fake_losses.append(to_numpy(fake_loss))

            if write_graph:
                # TODO: Uncomment when this works in tensorboardX
                # writer.add_graph(net_d, (pos_seq, adjs))
                # writer.add_graph(net_g, (noisev, adjs))
                write_graph = False

        # === Train generator ===
        freeze_net(net_d)
        for _ in range(args.g_iters):
            net_g.zero_grad()

            _, _, adjs = generate_batch(args, net_d, positive_rowgen)
            noise = generate_noise(net_g, adjs)
            noisev = Variable(noise.normal_(0, 1))
            gen_seq = net_g(noisev, adjs)
            gen_pred = net_d(gen_seq, adjs).sigmoid()
            gen_target = torch.ones(gen_pred.shape, device=settings.device)
            gen_loss = loss(gen_pred, gen_target)
            gen_loss.backward()
            del gen_seq

            optimizer_g.step()

            # Update stats
            if calculate_basic_statistics:
                stats.gen_preds.append(to_numpy(gen_pred))
                stats.gen_losses.append(to_numpy(gen_loss))

        # === Write Statistics ===
        if calculate_basic_statistics:
            resume_checkpoint = args.array_id and step == checkpoint.get("step")

            # Basic statistics
            with torch.no_grad(), eval_net(net_d), eval_net(net_g):
                stats.calculate_statistics_basic()

            # Extended statistics
            if calculate_extended_statistics:
                with torch.no_grad(), eval_net(net_d), eval_net(net_g):
                    stats.calculate_statistics_extended(net_d, net_g, internal_validation_datasets)

            # Write to disk
            if resume_checkpoint:
                validate_checkpoint(checkpoint, stats.scores)
            else:
                stats.write()
                write_checkpoint(args, stats, net_d, net_g)
                if current_performance is not None:
                    current_performance.update(writer.scalar_dict)

        stats.update()
        progressbar.update()


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    # === Arguments ===
    args = Args.from_cli()

    if args.gpu == -1:
        settings.device = torch.device("cpu")
        logger.info("Running on the CPU.")
    else:
        init_gpu(args.gpu)

    # === Random Seed ===
    random.seed(42 + args.array_id)
    np.random.seed(42 + args.array_id)
    torch.manual_seed(42 + args.array_id)
    random_state = np.random.RandomState(42)

    # === Paths ===
    tensorboard_path = args.root_path.joinpath("tensorboard")
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    # === Checkpoint ===
    checkpoint = load_checkpoint(args)

    # === Training Dataset ===
    positive_rowgen, negative_ds_gen = get_training_datasets(
        args, args.training_data_path, random_state
    )

    # === Internal Validation Dataset ===
    internal_validation_datasets = get_internal_validation_datasets(args, args.validation_data_path)

    # === Train ===
    start_time = time.perf_counter()
    result: Dict[str, Union[str, float]] = {}
    writer = SummaryWriter(tensorboard_path.as_posix())
    try:
        train(
            args,
            writer,
            positive_rowgen,
            negative_ds_gen,
            internal_validation_datasets,
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
