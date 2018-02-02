import argparse
import datetime
import json
import logging
import os
import os.path as op
import time
from pathlib import Path
from typing import Callable, List, NewType, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
# from memory_profiler import profile
# from line_profiler import LineProfiler
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import pagnn

logger = logging.getLogger(__name__)

# profile = LineProfiler()

DataSet = NewType('DataSet', Tuple[Variable, Variable, Variable])


def calculate_score(targets: np.ndarray, outputs: np.ndarray) -> float:
    score = metrics.roc_auc_score(targets, outputs)
    return score


def evaluate_validation_dataset(net, datagen):
    """Evaluate the performance of the network on the validation subset.

    Closure with `net` in its scope.
    """
    outputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for idx, (pos, neg) in enumerate(datagen()):
        outputs = net(pos, neg)
        targets = Variable(torch.FloatTensor([1] + [0] * len(neg)).cuda()).unsqueeze(1)
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
        outputs = net(seq, adjs)
        outputs_arr = pagnn.to_numpy(outputs)
        outputs_arr_diff = outputs_arr[1::2] - outputs_arr[0::2]
        targets_arr = pagnn.to_numpy(targets)
        assert outputs_arr_diff.shape == targets_arr.shape
        outputs_list.append(outputs_arr_diff)
        targets_list.append(targets_arr)
    outputs = np.hstack(outputs_list)
    targets = np.hstack(targets_list)
    return targets, outputs


def get_log_dir(args) -> str:

    def str_(f):
        return str(f).replace('.', '_')

    log_dir = '-'.join([
        args.network_name, args.loss_name,
        str_(args.lr),
        str_(args.weight_decay),
        str_(args.n_filters),
        pagnn.__version__.replace('.', '_'),
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    ])

    return log_dir


# @profile
def main(args: argparse.Namespace,
         start_idx: int = 0,
         training_datagen: Callable[[], DataSet],
         internal_validation_datagen: Callable[[], DataSet],
         batch_size=256,
         validation_step_size=10_000,
         _num_aa_to_process=None):
    """"""
    # Network
    net = getattr(pagnn.models, args.network_name)(n_filters=args.n_filters)
    if torch.cuda.is_available():
        net.cuda()
    criterion = getattr(nn, args.loss_name)()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Logs
    unique_name = get_log_dir(args)
    tensorboard_runs_path = working_path.joinpath(unique_name).joinpath('tensorboard')
    tensorboard_runs_path.mkdir(parents=True, exist_ok=True)
    model_dump_path = working_path.joinpath(unique_name).joinpath('models')
    model_dump_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tensorboard_runs_path.as_posix())
    # Train
    targets: List[Variable] = []
    outputs: List[Variable] = []
    num_aa_processed = 0
    for idx, (pos, neg) in enumerate(tqdm.tqdm(training_datagen())):
        try:
            if start_idx or (idx % 10_000 == 0):
                logger.debug("Calculating score...")
                # Not supported on PyTorch version < 0.4
                # if idx == 0:
                #     model_filename = op.join(writer.file_writer.get_logdir(), 'model.proto')
                #     torch.onnx.export(net, (seq, adjs), model_filename, verbose=True)
                #     writer.add_graph_onnx(model_filename)

                # x = vutils.make_grid(net.spatial_conv, normalize=True, scale_each=True)
                # writer.add_image('Spatial convolutions', x, idx)

                score = metrics.roc_auc_score(
                    pagnn.to_numpy(targets_batch), pagnn.to_numpy(outputs_batch))

                targets_valid, outputs_valid = evaluate_validation_dataset(
                    net, internal_validation_datagen)
                # nan_mask = np.isnan(outputs_valid)
                # if nan_mask.any():
                #     logger.warning("Network returned %s nans on the validation dataset!",
                #                    nan_mask.sum())
                #     targets_valid = targets_valid[~nan_mask]
                #     outputs_valid = outputs_valid[~nan_mask]
                score_valid = metrics.roc_auc_score(targets_valid, outputs_valid)

                # Write parameters
                for name, param in net.named_parameters():
                    writer.add_histogram(name, pagnn.to_numpy(param), idx)

                writer.add_histogram('outputs', pagnn.to_numpy(outputs_batch), idx)
                writer.add_histogram('outputs_valid', outputs_valid, idx)

                writer.add_scalar('score', score, idx)
                writer.add_scalar('score_valid', score_valid, idx)
                writer.add_scalar('num_aa_processed', num_aa_processed, idx)

                writer.add_pr_curve('Training score', pagnn.to_numpy(targets_batch),
                                    pagnn.to_numpy(outputs_batch), idx)
                writer.add_pr_curve('Validation score', targets_valid, outputs_valid, idx)

                logger.debug("Serializing trained network...")
                result = writer.scalar_dict.copy()
                info_file = working_path.joinpath(f'info.json')
                info_file2 = working_path.joinpath(f'info-step{idx}.json')
                model_path = model_dump_path.joinpath(f'model-step{idx}')

                result['model_folder'] = model_path.name
                torch.save(net.state_dict(), model_path.as_posix())
                # Otherwise, make sure that the data we got matches

            target = Variable(torch.FloatTensor([1] + [0] * len(neg)).cuda()).unsqueeze(1)
            targets.extend(target)
            output = net(pos, neg)
            outputs.extend(output)
            # loss = criterion(output, target)
            # losses.append(loss)
            num_aa_processed += pos[0][0].size()[1]

            if idx % 256 == 0:
                logger.debug("Updating network...")
                targets_batch = torch.cat(targets)
                outputs_batch = torch.cat(outputs)
                targets = []
                outputs = []

                loss = criterion(outputs_batch, targets_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.num_aa_to_process is not None and num_aa_processed >= args.num_aa_to_process:
                break
        except KeyboardInterrupt:
            break
    result = writer.scalar_dict.copy()
    writer.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-concurrent-jobs', type=int, default=1)

    parser.add_argument('--workdir', type=str, default='.')
    parser.add_argument('--cachedir', type=str, default='.cache')

    parser.add_argument('--num-aa-to-process', type=int, default=None)
    parser.add_argument('--max-seq-length', type=int, default=50_000)
    parser.add_argument('--network_name', type=str, default='ModernNet')
    parser.add_argument('--loss_name', type=str, default='BCELoss')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--n_filters', type=int, default=12)
    parser.add_argument('--working-path', type=str)

    parser.add_argument('--nocuda',)
    parser.add_argument('--min-seq-identity', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Load previosly calculated parameters
    args_dict = vars(args)

    working_path = Path(args.working_path).absolute()

    # === Training ===
    logger.info("Setting up training datagen...")
    training_datagen = pagnn.get_training_datagen(working_path, args.min_seq_identity)

    # === Validation ===
    logger.info("Setting up validation datagen...")
    validation_datagen = pagnn.get_validation_datagen(working_path)

    # === Network ===
    start_time = time.perf_counter()
    pagnn.init_gpu()
    result = main(args, training_datagen, validation_datagen)
    time_elapsed = time.perf_counter() - start_time

    print(json.dumps(result))
