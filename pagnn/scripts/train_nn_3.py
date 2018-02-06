import argparse
import hashlib
import itertools
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NewType, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from scipy import sparse
# from memory_profiler import profile
# from line_profiler import LineProfiler
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import pagnn
from pagnn import DataSet, DataSetCollection

logger = logging.getLogger(__name__)

# profile = LineProfiler()


def calculate_score(targets: np.ndarray, outputs: np.ndarray) -> float:
    score = metrics.roc_auc_score(targets, outputs)
    return score


def evaluate_validation_dataset(net,
                                datagen,
                                keep_neg_seq=False,
                                keep_neg_adj=False,
                                fake_adj=False):
    """Evaluate the performance of the network on the validation subset.

    Closure with `net` in its scope.
    """
    assert keep_neg_seq or keep_neg_adj
    if fake_adj:
        assert keep_neg_adj and not keep_neg_seq

    outputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for idx, (pos, neg) in enumerate(datagen()):
        if fake_adj:
            for i in range(len(neg)):
                neg_ds = neg[i]
                data = np.ones(len(neg_ds.seq))
                row = np.arange(len(neg_ds.seq))
                neg[i] = DataSet(neg_ds.seq, sparse.coo_matrix((data, (row, row))), neg_ds.target)
        dvc = pagnn.push_dataset_collection((pos, neg), keep_neg_seq, keep_neg_adj)
        targets = pagnn.get_training_targets(dvc)
        outputs = net(dvc)
        outputs_list.append(pagnn.to_numpy(outputs))
        targets_list.append(pagnn.to_numpy(targets).astype(int))
    # import pdb; pdb.set_trace()
    outputs = np.hstack(outputs_list)
    targets = np.hstack(targets_list)
    return targets, outputs


def evaluate_mutation_dataset(net, datagen):
    outputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for idx, (seq, adjs, targets) in enumerate(datagen()):
        outputs = net((seq, adjs))
        outputs_arr = pagnn.to_numpy(outputs)
        outputs_arr_diff = outputs_arr[1::2] - outputs_arr[0::2]
        targets_arr = pagnn.to_numpy(targets)
        assert outputs_arr_diff.shape == targets_arr.shape
        outputs_list.append(outputs_arr_diff)
        targets_list.append(targets_arr)
    outputs = np.hstack(outputs_list)
    targets = np.hstack(targets_list)
    return targets, outputs


# @profile
def main(args: argparse.Namespace,
         work_path: Path,
         writer: SummaryWriter,
         training_datagen: Callable[[], Iterator[DataSetCollection]],
         internal_validation_datagens: Dict[str, Callable[[], Iterator[DataSetCollection]]],
         mutation_validation_datagens: Dict[str, Callable[[], Iterator[DataSetCollection]]],
         batch_size=256,
         steps_between_validation=10_240,
         current_performance: Optional[dict] = None):
    """"""
    assert steps_between_validation % batch_size == 0

    models_path = work_path.joinpath('models')
    models_path.mkdir(exist_ok=True)

    info_file = work_path.joinpath('info.json')
    checkpoint_file = work_path.joinpath('checkpoint.json')

    # Set up network
    net = getattr(pagnn.models, args.network_name)(n_filters=args.n_filters)
    if pagnn.CUDA:
        net = net.cuda()
    criterion = getattr(nn, args.loss_name)()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load checkpoint
    args_dict = vars(args)
    checkpoint: dict = {}
    if args.resume:
        with info_file.open('rt') as fin:
            info = json.load(fin)
        assert all(info[k] == args_dict[k] for k in args_dict if k not in ['resume'])
        with checkpoint_file.open('rt') as fin:
            checkpoint = json.load(fin)
        assert checkpoint['unique_name'] == work_path.name
    else:
        if info_file.is_file():
            raise Exception(f"Info file '{info_file}' already exists!")
        args_dict = vars(args)
        with info_file.open('wt') as fout:
            json.dump(args_dict, fout, sort_keys=True, indent=4)

    # Load network from checkpoint
    if checkpoint:
        net.load_state_dict(
            torch.load(models_path.joinpath(checkpoint['model_path_name']).as_posix()))

    # Train
    targets: List[Variable] = []
    outputs: List[Variable] = []
    num_aa_processed = 0
    for step, (pos, neg) in enumerate(
            tqdm.tqdm(training_datagen()), start=checkpoint.get('step', 0)):

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
                options = [('neg_seq', True, False, False), ('neg_adj', False, True, False),
                           ('neg_control', False, True, True)]
                for suffix, keep_neg_seq, keep_neg_adj, fake_adj in options:
                    if '_permute_' in validation_name and suffix == 'neg_adj':
                        # 'permute' method does not generate negative adjacencies
                        continue
                    targets_valid, outputs_valid = evaluate_validation_dataset(
                        net, validation_datagen, keep_neg_seq, keep_neg_adj, fake_adj)
                    scores[f'{validation_name}-{suffix}'] = metrics.roc_auc_score(
                        targets_valid, outputs_valid)

            # === Write ===
            if step == checkpoint.get('step'):
                logger.debug('Validating checkpoint.')
                assert all(checkpoint[s] == scores[s] for s in scores)
            else:
                logger.debug('Saving checkpoint.')
                for name, param in net.named_parameters():
                    writer.add_histogram(name, pagnn.to_numpy(param), step)

                for score_name, score_value in scores.items():
                    writer.add_scalar(score_name, score_value, step)

                writer.add_scalar('num_aa_processed', num_aa_processed, step)

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
        target = Variable(torch.FloatTensor([1] + [0] * len(neg)).cuda()).unsqueeze(1)
        targets.extend(target)
        output = net((pos, neg))
        outputs.extend(output)

        # Update statistics
        num_aa_processed += pos[0][0].size()[1]

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
    parser.add_argument('--network_name', type=str, default='ModernNet')
    parser.add_argument('--loss_name', type=str, default='BCELoss')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--n_filters', type=int, default=12)
    # Training set arguments
    parser.add_argument('--training-methods', type=str, default='permute')
    parser.add_argument('--training-min-seq-identity', type=int, default=0)
    parser.add_argument('--training-permutes', default='seq', choices=['seq', 'adj', 'both'])
    # Validation set arguments
    parser.add_argument('--validation-methods', type=str, default='exact')
    parser.add_argument('--validation-num-sequences', type=int, default=10_000)
    parser.add_argument('--validation-min-seq-identity', type=int, default=80)
    # Other things to process
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--num-aa-to-process', type=int, default=None)
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
        Path(__file__).stem, args.training_methods,
        str(args.training_min_seq_identity), state_hash
    ])
    return log_dir


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Arguments
    args = parse_args()
    if args.no_gpu:
        pagnn.CUDA = False

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
    training_datagen = pagnn.get_datagen('training', data_path, args.training_min_seq_identity,
                                         args.training_methods.split('.'))

    # === Internal Validation ===
    logger.info("Setting up validation datagen...")
    validation_datagens = {}
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
            datagen = pagnn.get_datagen('validation', data_path, 80, [method],
                                        np.random.RandomState(42))()
            dataset = list(
                tqdm.tqdm(
                    itertools.islice(datagen, args.validation_num_sequences),
                    total=args.validation_num_sequences,
                    desc=cache_file.name))
            assert len(dataset) == args.validation_num_sequences
            with cache_file.open('wb') as fout:
                pickle.dump(dataset, fout, pickle.HIGHEST_PROTOCOL)
        validation_datagens[datagen_name] = lambda dataset=dataset: dataset

    # === Mutation Validation ===
    # TODO

    # === Train ===
    pagnn.init_gpu()
    start_time = time.perf_counter()
    result: dict = {}
    writer = SummaryWriter(tensorboard_path.as_posix())
    try:
        main(
            args,
            work_path,
            writer,
            training_datagen,
            validation_datagens, {},
            current_performance=result)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
    result['time_elapsed'] = time.perf_counter() - start_time

    # === Output ===
    print(json.dumps(result, sort_keys=True, indent=4))
