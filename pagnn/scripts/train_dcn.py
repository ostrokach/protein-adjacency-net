import argparse
import hashlib
import itertools
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from scipy import stats
# from memory_profiler import profiles
# from line_profiler import LineProfiler
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import pagnn
from pagnn import settings
from pagnn.scripts._train_dcn import (evaluate_mutation_dataset, evaluate_validation_dataset,
                                      get_datagen, get_mutation_datagen, push_dataset_collection)
from pagnn.types import DataGen, DataSetCollection

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace,
         work_path: Path,
         writer: SummaryWriter,
         training_datagen: Callable[[], Iterator[DataSetCollection]],
         internal_validation_datagens: Mapping[str, DataGen],
         mutation_validation_datagens: Mapping[str, DataGen],
         batch_size=256,
         steps_between_validation=25_600,
         current_performance: Optional[Dict[str, Union[str, float]]] = None):
    """"""
    assert steps_between_validation % batch_size == 0

    models_path = work_path.joinpath('models')
    models_path.mkdir(exist_ok=True)

    info_file = work_path.joinpath('info.json')
    checkpoint_file = work_path.joinpath('checkpoint.json')

    # Set up network
    net = getattr(pagnn.models, args.network_name)(n_filters=args.n_filters)
    if pagnn.settings.CUDA:
        net = net.cuda()
    criterion = getattr(nn, args.loss_name)()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load checkpoint
    args_dict = vars(args)
    checkpoint: dict = {}
    if args.resume:
        with info_file.open('rt') as fin:
            info = json.load(fin)
        for key in info:
            if key in ['resume']:
                continue
            if info[key] != args_dict.get(key):
                logger.warning("The value for parameter '%s' is different from the previous run. "
                               "('%s' != '%s')", key, info[key], args_dict.get(key))
        with checkpoint_file.open('rt') as fin:
            checkpoint = json.load(fin)
        assert checkpoint['unique_name'] == work_path.name
        assert checkpoint['step'] > 0
        net.load_state_dict(
            torch.load(models_path.joinpath(checkpoint['model_path_name']).as_posix()))
    else:
        if info_file.is_file():
            raise Exception(f"Info file '{info_file}' already exists!")
        args_dict = vars(args)
        with info_file.open('wt') as fout:
            json.dump(args_dict, fout, sort_keys=True, indent=4)

    # Train
    targets: List[Variable] = []
    outputs: List[Variable] = []
    num_aa_processed = 0
    validation_time = None
    for step, (pos, neg) in enumerate(
            tqdm.tqdm(
                training_datagen(),
                initial=checkpoint.get('step', 0),
                disable=not settings.SHOW_PROGRESSBAR),
            start=checkpoint.get('step', 0)):

        # Validation score
        if step % steps_between_validation == 0:
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
                scores['training'] = metrics.roc_auc_score(
                    pagnn.to_numpy(torch.cat(targets)), pagnn.to_numpy(torch.cat(outputs)))

            # Validation
            for validation_name, validation_datagen in internal_validation_datagens.items():
                options = [('seq', True, False, False), ('adj', False, True, False), ('zzz', True,
                                                                                      False, True)]
                for suffix, keep_neg_seq, keep_neg_adj, fake_adj in options:
                    if '_permute_' in validation_name and suffix == 'adj':
                        # 'permute' method does not generate negative adjacencies
                        continue
                    targets_valid, outputs_valid = evaluate_validation_dataset(
                        net, validation_datagen, keep_neg_seq, keep_neg_adj, fake_adj)
                    scores[f'{validation_name}-{suffix}'] = metrics.roc_auc_score(
                        targets_valid, outputs_valid)

            for name, datagen in external_validation_datagens.items():
                targets_valid, outputs_valid = evaluate_mutation_dataset(net, datagen)
                if 'protherm' in name:
                    # Protherm predicts ΔΔG, so positive values are destabilizing
                    scores[name + '-spearman_r'] = stats.spearmanr(-targets_valid,
                                                                   outputs_valid).correlation
                elif 'humsavar' in name:
                    # For humsavar: 0 = stable, 1 = deleterious
                    scores[name + '-auc'] = metrics.roc_auc_score(1 - targets_valid, outputs_valid)
                else:
                    scores[name] = metrics.roc_auc_score(targets_valid + 1, outputs_valid)

            # === Write ===
            if args.resume and step == checkpoint.get('step'):
                logger.debug('Validating checkpoint.')
                common_scores = set(checkpoint) & set(scores)
                assert common_scores
                assert all(checkpoint[s] == scores[s] for s in common_scores)
            else:
                logger.debug('Saving checkpoint.')
                for name, param in net.named_parameters():
                    writer.add_histogram(name, pagnn.to_numpy(param), step)

                for score_name, score_value in scores.items():
                    writer.add_scalar(score_name, score_value, step)

                writer.add_scalar('num_aa_processed', num_aa_processed, step)

                prev_validation_time = validation_time
                validation_time = time.perf_counter()
                if prev_validation_time is not None:
                    sequences_per_second = (steps_between_validation /
                                            (validation_time - prev_validation_time))
                    writer.add_scalar('sequences_per_second', sequences_per_second, step)

                if outputs:
                    writer.add_histogram('outputs', pagnn.to_numpy(torch.cat(outputs)), step)
                    writer.add_pr_curve('Training', pagnn.to_numpy(torch.cat(targets)),
                                        pagnn.to_numpy(torch.cat(outputs)), step)
                # writer.add_histogram('outputs_valid', outputs_valid, step)
                # writer.add_pr_curve('Validation', targets_valid, outputs_valid, step)

                # Save model
                model_dump_path = models_path.joinpath(f'step-{step}.model')
                torch.save(net.state_dict(), model_dump_path.as_posix())

                # Save checkpoint
                checkpoint = {
                    'step': step,
                    'unique_name': unique_name,
                    'model_path_name': model_dump_path.name,
                    **scores,
                }
                with work_path.joinpath(f'checkpoint-step{step}.json').open('wt') as fout:
                    json.dump(checkpoint, fout, sort_keys=True, indent=4)
                with checkpoint_file.open('wt') as fout:
                    json.dump(checkpoint, fout, sort_keys=True, indent=4)

                if current_performance is not None:
                    current_performance.update(writer.scalar_dict)

            # === Reset parameters ===
            validation_time = time.perf_counter()

        # Update network
        if (step % batch_size == 0) and outputs:
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
        dvc, target = push_dataset_collection((pos, neg), 'seq' in args.training_permutations,
                                              'adj' in args.training_permutations)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Paths
    # # Location where to create subfolders for storing network data and cache files.
    parser.add_argument('--rootdir', type=str, default='.')
    # # Location of the `adjacency-net` databin folder.
    parser.add_argument('--datadir', type=str, default='.')
    # Network parameters
    parser.add_argument('--network_name', type=str, default='Classifier')
    parser.add_argument('--loss_name', type=str, default='BCELoss')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--n_filters', type=int, default=64)
    # Training set arguments
    parser.add_argument('--training-methods', type=str, default='permute')
    parser.add_argument('--training-min-seq-identity', type=int, default=0)
    parser.add_argument('--training-permutations', default='seq', choices=['seq', 'adj', 'seq.adj'])
    # Validation set arguments
    parser.add_argument(
        '--validation-methods', type=str, default='permute.start.stop.middle.edges.exact')
    parser.add_argument('--validation-num-sequences', type=int, default=10_000)
    parser.add_argument('--validation-min-seq-identity', type=int, default=80)
    # Other things to process
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=pagnn.settings.ARRAY_JOB)
    parser.add_argument('--num-aa-to-process', type=int, default=None)
    # Visuals
    parser.add_argument('--progress', action='store_true', default=pagnn.settings.SHOW_PROGRESSBAR)
    # TODO(AS):
    parser.add_argument('-n', '--num-concurrent-jobs', type=int, default=1)
    args = parser.parse_args()
    return args


def get_log_dir(args) -> str:
    args_dict = vars(args)
    state_keys = ['network_name', 'loss_name', 'lr', 'weight_decay', 'n_filters']
    state_dict = {k: args_dict[k] for k in state_keys}
    state_dict['pagnn_version'] = pagnn.__version__
    # https://stackoverflow.com/a/22003440/2063031
    state_hash = hashlib.md5(json.dumps(state_dict, sort_keys=True).encode('ascii')).hexdigest()
    log_dir = '-'.join([
        Path(__file__).stem, args.training_methods, args.training_permutations,
        str(args.training_min_seq_identity), state_hash
    ] + ([args.tag] if args.tag else []))
    return log_dir


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Arguments
    args = parse_args()

    if args.gpu == -1:
        pagnn.settings.CUDA = False
        logger.info("Running on the CPU.")
    else:
        pagnn.init_gpu(args.gpu)

    # === Paths ===
    root_path = Path(args.rootdir).absolute()
    data_path = Path(args.datadir).absolute()
    cache_path = data_path

    unique_name = get_log_dir(args)
    work_path = root_path.joinpath(unique_name)
    tensorboard_path = work_path.joinpath('tensorboard')
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    # === Training ===
    logger.info("Setting up training datagen...")
    training_datagen = get_datagen('training', data_path, args.training_min_seq_identity,
                                   args.training_methods.split('.'))

    # === Internal Validation ===
    logger.info("Setting up validation datagen...")
    internal_validation_datagens: Dict[str, DataGen] = {}
    min_seq_identity = 80
    for method in args.validation_methods.split('.'):
        datagen_name = (f'validation_{method}_{args.validation_min_seq_identity}'
                        f'_{args.validation_num_sequences}')
        cache_file = root_path.joinpath(datagen_name + '.pickle')
        try:
            with cache_file.open('rb') as fin:
                dataset = pickle.load(fin)
            logger.info("Loaded validation datagen from file: '%s'.", cache_file)
            assert len(dataset) == args.validation_num_sequences
        except FileNotFoundError:
            logger.info("Generating validation datagen: '%s'.", datagen_name)
            datagen: DataGen = get_datagen('validation', data_path, 80, [method],
                                           np.random.RandomState(42))
            dataset = list(
                tqdm.tqdm(
                    itertools.islice(datagen(), args.validation_num_sequences),
                    total=args.validation_num_sequences,
                    desc=cache_file.name,
                    disable=not settings.SHOW_PROGRESSBAR))
            assert len(dataset) == args.validation_num_sequences
            with cache_file.open('wb') as fout:
                pickle.dump(dataset, fout, pickle.HIGHEST_PROTOCOL)

        def datagen_from_memory() -> Iterator[DataSetCollection]:
            return dataset

        internal_validation_datagens[datagen_name] = datagen_from_memory

    # === Mutation Validation ===
    external_validation_datagens: Dict[str, DataGen] = {}
    for mutation_class in ['protherm', 'humsavar']:
        external_validation_datagens[f'validation_{mutation_class}'] = get_mutation_datagen(
            mutation_class, data_path)

    # === Train ===
    start_time = time.perf_counter()
    result: Dict[str, Union[str, float]] = {}
    writer = SummaryWriter(tensorboard_path.as_posix())
    try:
        main(
            args,
            work_path,
            writer,
            training_datagen,
            internal_validation_datagens,
            external_validation_datagens,
            current_performance=result)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
    result['time_elapsed'] = time.perf_counter() - start_time

    # === Output ===
    print(json.dumps(result, sort_keys=True, indent=4))
