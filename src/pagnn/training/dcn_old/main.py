import json
import logging
import random
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable

import pagnn
from pagnn import settings
from pagnn.datavardcn import push_dataset_collection
from pagnn.training.dcn_old import evaluate_validation_dataset, get_datagen
from pagnn.types import DataSetCollection

from .args import Args

logger = logging.getLogger(__name__)


def train(
    args: Args,
    writer,
    training_datagen: Callable[[], Iterator[DataSetCollection]],
    negative_dataset_gen,
    internal_validation_datagens,
    checkpoint: Dict[str, Any],
    current_performance: Optional[Dict[str, Union[str, float]]] = None,
):
    """"""
    models_path = args.work_path.joinpath("models")
    models_path.mkdir(exist_ok=True)

    info_file = args.work_path.joinpath("info.json")
    checkpoint_file = args.work_path.joinpath("checkpoint.json")

    # Set up network
    net = getattr(pagnn.models, args.network_name)(**args.network_settings).to(settings.device)

    criterion = getattr(nn, args.loss_name)()

    optimizer = optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Load checkpoint
    args_dict = args.to_dict()
    if args.resume:
        with info_file.open("rt") as fin:
            info = json.load(fin)
        for key in info:
            if key in ["resume"]:
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
        assert checkpoint["unique_name"] == args.work_path.name
        assert checkpoint["step"] > 0
        net.load_state_dict(
            torch.load(models_path.joinpath(checkpoint["model_path_name"]).as_posix())
        )
    else:
        if info_file.is_file():
            raise Exception(f"Info file '{info_file}' already exists!")
        args_dict = vars(args)
        with info_file.open("wt") as fout:
            json.dump(args_dict, fout, sort_keys=True, indent=4)

    # Train
    targets: List[Variable] = []
    outputs: List[Variable] = []
    num_aa_processed = 0
    validation_time = None
    for step, (pos, neg) in training_datagen():

        # Validation score
        if step % args.steps_between_validation == 0:
            logger.debug("Calculating score...")

            # Not supported on PyTorch version < 0.4
            # if idx == 0:
            #     model_filename = op.join(writer.file_writer.get_logdir(), 'model.proto')
            #     torch.onnx.export(net, (seq, adjs), model_filename, verbose=True)
            #     writer.add_graph_onnx(model_filename)

            # x = vutils.make_grid(net.spatial_conv, normalize=True, scale_each=True)
            # writer.add_image('Spatial convolutions', x, idx)

            # === Evaluate ===
            scores = {}

            # Training
            if outputs:
                scores["training"] = metrics.roc_auc_score(
                    torch.cat(targets).numpy(), torch.cat(outputs).numpy()
                )

            # Validation
            for validation_name, validation_datagen in internal_validation_datagens.items():
                options = [
                    ("seq", True, False, False),
                    ("adj", False, True, False),
                    ("zzz", True, False, True),
                ]
                for suffix, keep_neg_seq, keep_neg_adj, fake_adj in options:
                    if "_permute_" in validation_name and suffix == "adj":
                        # 'permute' method does not generate negative adjacencies
                        continue
                    targets_valid, outputs_valid = evaluate_validation_dataset(
                        net, validation_datagen, keep_neg_seq, keep_neg_adj, fake_adj
                    )
                    scores[f"{validation_name}-{suffix}"] = metrics.roc_auc_score(
                        targets_valid, outputs_valid
                    )

            # === Write ===
            if args.resume and step == checkpoint.get("step"):
                logger.debug("Validating checkpoint.")
                common_scores = set(checkpoint) & set(scores)
                assert common_scores
                assert all(checkpoint[s] == scores[s] for s in common_scores)
            else:
                logger.debug("Saving checkpoint.")
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.numpy(), step)

                for score_name, score_value in scores.items():
                    writer.add_scalar(score_name, score_value, step)

                writer.add_scalar("num_aa_processed", num_aa_processed, step)

                prev_validation_time = validation_time
                validation_time = time.perf_counter()
                if prev_validation_time is not None:
                    sequences_per_second = args.steps_between_validation / (
                        validation_time - prev_validation_time
                    )
                    writer.add_scalar("sequences_per_second", sequences_per_second, step)

                if outputs:
                    writer.add_histogram("outputs", torch.cat(outputs).numpy(), step)
                    writer.add_pr_curve(
                        "Training", torch.cat(targets).numpy(), torch.cat(outputs).numpy(), step
                    )
                # writer.add_histogram('outputs_valid', outputs_valid, step)
                # writer.add_pr_curve('Validation', targets_valid, outputs_valid, step)

                # Save model
                model_dump_path = models_path.joinpath(f"step-{step}.model")
                torch.save(net.state_dict(), model_dump_path.as_posix())

                # Save checkpoint
                checkpoint = {
                    "step": step,
                    "unique_name": args.unique_name,
                    "model_path_name": model_dump_path.name,
                    **scores,
                }
                with args.work_path.joinpath(f"checkpoint-step{step}.json").open("wt") as fout:
                    json.dump(checkpoint, fout, sort_keys=True, indent=4)
                with checkpoint_file.open("wt") as fout:
                    json.dump(checkpoint, fout, sort_keys=True, indent=4)

                if current_performance is not None:
                    current_performance.update(writer.scalar_dict)

            # === Reset parameters ===
            validation_time = time.perf_counter()

        # Update network
        if (step % args.batch_size == 0) and outputs:
            logger.debug("Updating network...")
            loss = criterion(torch.cat(outputs), torch.cat(targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Reset variables
            outputs = []
            targets = []

        # Step through network
        # TODO: Weigh positive and negative examples differently
        # weights = pagnn.get_training_weights((pos, neg))
        dvc, target = push_dataset_collection(
            (pos, neg), "seq" in args.training_permutations, "adj" in args.training_permutations
        )
        targets.extend(target)
        output = net(dvc)
        outputs.extend(output)

        # Update statistics
        num_aa_processed += sum(len(ds.seq) for ds in pos)

        # Stopping criterion
        if args.num_aa_to_process is not None and num_aa_processed >= args.num_aa_to_process:
            break

    writer.close()
    return current_performance


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    args = Args.from_cli()

    if args.gpu == -1:
        pagnn.settings.device = torch.device("cpu")
        logger.info("Running on the CPU.")
    else:
        pagnn.init_gpu(args.gpu)

    random.seed(42 + args.array_id)
    np.random.seed(42 + args.array_id)
    torch.manual_seed(42 + args.array_id)
    random_state = np.random.RandomState(42 + args.array_id)

    tensorboard_path = args.work_path.joinpath("tensorboard")
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    # === Training ===
    logger.info("Setting up training datagen...")
    training_datagen = get_datagen(args.data_path, args.training_methods.split("."), random_state)

    # === Train ===
    start_time = time.perf_counter()
    result: Dict[str, Union[str, float]] = {}
    writer = tensorboard_path
    try:
        main(args, writer, training_datagen, current_performance=result)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
    result["time_elapsed"] = time.perf_counter() - start_time

    # === Output ===
    print(json.dumps(result, sort_keys=True, indent=4))
