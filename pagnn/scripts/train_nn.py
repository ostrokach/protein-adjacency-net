import json
import logging
import os
import os.path as op
import pickle
import random
import time
from pathlib import Path
from typing import Callable, List, NewType, Tuple

import GPUtil
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


def init_gpu():
    """Select the least active GPU."""
    deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, maxMemory=0.5)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in deviceIDs)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def init_network(network_name='MultiDomainNet',
                 loss_name='BCELoss',
                 lr=0.01,
                 weight_decay=0.001,
                 n_filters=12):
    net = getattr(pagnn.models, network_name)(n_filters=n_filters)
    if torch.cuda.is_available():
        net.cuda()

    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # BCELoss
    criterion = getattr(nn, loss_name)()
    # criterion = nn.BCEWithLogitsLoss()

    # optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.1)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)

    def str_(f):
        return str(f).replace('.', '_')

    log_dir = '-'.join([network_name, loss_name, str_(lr), str_(weight_decay), str_(n_filters)])
    writer = SummaryWriter(op.join('runs', log_dir))

    return net, criterion, optimizer, writer


def to_np(x: Variable):
    """Convert PyTorch `Variable` to numpy array."""
    return x.data.cpu().numpy()


def to_var(x: np.ndarray):
    """Convert numpy array to PyTorch `Variable`."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def step_through_network(net,
                         criterion,
                         optimizer,
                         seq_var,
                         adjs_expanded_var,
                         targets_var,
                         calc_score=False):
    """Perform full pass over network.

    Closure with `net`, `criterion`, and `optimizer` in its scope.
    """
    optimizer.zero_grad()
    output = net(seq_var, adjs_expanded_var)
    loss = criterion(output, targets_var)
    loss.backward()
    optimizer.step()
    return output


def calculate_score(targets: np.ndarray, outputs: np.ndarray) -> float:
    score = metrics.roc_auc_score(targets, outputs)
    return score


def evaluate_validation_dataset(net, datagen):
    """Evaluate the performance of the network on the validation subset.

    Closure with `net` in its scope.
    """
    outputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for idx, (seq, adjs, targets) in enumerate(datagen()):
        outputs = net(seq, adjs)
        outputs_list.append(to_np(outputs))
        targets_list.append(to_np(targets).astype(int))
    outputs = np.hstack(outputs_list)
    targets = np.hstack(targets_list)
    return targets, outputs


def evaluate_humsavar(net, datasets):
    ...


def evaluate_protherm():
    ...


def get_pytorch_vars(seq, adjs, targets=None):
    seq_permuted = pagnn.permute_sequence(seq)
    seq_ar = np.array([pagnn.get_seq_array(seq), pagnn.get_seq_array(seq_permuted)])
    seq_var = Variable(torch.Tensor(seq_ar)).cuda()

    adjs_expanded = [pagnn.expand_adjacency(adj) for adj in adjs]
    adjs_expanded_var = [
        Variable(torch.FloatTensor(adj.astype(np.float32))).cuda() for adj in adjs_expanded
    ]

    if targets is None:
        targets = [1, 0] * len(adjs)
        targets_var = Variable(torch.Tensor(np.array(targets, dtype=np.float32))).cuda()

    return seq_var, adjs_expanded_var, targets_var


# @profile
def main(net,
         criterion,
         optimizer,
         writer,
         training_datagen: Callable[[], DataSet],
         internal_validation_datagen: Callable[[], DataSet],
         external_validation_datagen: Callable[[], DataSet],
         _num_aa_to_process=None):
    """"""
    num_aa_processed = 0
    for idx, (seq, adjs, targets) in enumerate(tqdm.tqdm(training_datagen())):
        try:
            outputs = step_through_network(net, criterion, optimizer, seq, adjs, targets)

            calc_score = idx % 100 == 0
            if calc_score:
                # Not supported on PyTorch version < 0.4
                # if idx == 0:
                #     model_filename = op.join(writer.file_writer.get_logdir(), 'model.proto')
                #     torch.onnx.export(net, (seq, adjs), model_filename, verbose=True)
                #     writer.add_graph_onnx(model_filename)

                # x = vutils.make_grid(net.spatial_conv, normalize=True, scale_each=True)
                # writer.add_image('Spatial convolutions', x, idx)

                for name, param in net.named_parameters():
                    writer.add_histogram(name, to_np(param), idx)

                score = calculate_score(to_np(targets), to_np(outputs))

                targets_valid, outputs_valid = evaluate_validation_dataset(
                    net, internal_validation_datagen)
                score_valid = calculate_score(targets_valid, outputs_valid)

                writer.add_scalar('score', score, idx)
                writer.add_scalar('score_valid', score_valid, idx)
                writer.add_scalar('num_aa_processed', num_aa_processed, idx)

                writer.add_pr_curve('Training score', to_np(targets), to_np(outputs), idx)
                writer.add_pr_curve('Validation score', targets_valid, outputs_valid, idx)

            num_aa_processed += seq.size()[2]
            if _num_aa_to_process is not None and num_aa_processed >= _num_aa_to_process:
                break
        except KeyboardInterrupt:
            break
    result = writer.scalar_dict.copy()
    writer.close()
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-aa-to-process', type=int, default=None)
    parser.add_argument('-m', '--max-seq-length', type=int, default=50_000)
    parser.add_argument('--net', type=str, default='MultiDomainNet')
    parser.add_argument('--loss', type=str, default='BCELoss')
    parser.add_argument('--lr', type=int, default=50_000)
    parser.add_argument('--weight_decay', type=int, default=50_000)
    parser.add_argument('--n_filters', type=int, default=12)
    parser.add_argument('--working-path', type=str)
    args = parser.parse_args()

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    random.seed(42)
    np.random.seed(42)

    working_path = Path(args.working_path).absolute()

    training_domain_folders_file = working_path.joinpath('training_domain_folders.pickle')
    with training_domain_folders_file.open('rb') as fin:
        training_domain_folders = pickle.load(fin)

    training_domain_weights_file = working_path.joinpath('training_domain_weights.pickle')
    with training_domain_weights_file.open('rb') as fin:
        training_domain_weights = pickle.load(fin)

    validation_datasets_file = working_path.joinpath('validation_datasets.pickle')
    with validation_datasets_file.open('rb') as fin:
        validation_datasets = pickle.load(fin)

    def training_datagen():
        yield from (get_pytorch_vars(seq, adjs)
                    for seq, adjs in pagnn.iter_datasets(
                        pagnn.iter_dataset_rows(training_domain_folders, training_domain_weights),
                        args.max_seq_length))

    def internal_validation_datagen():
        yield from (get_pytorch_vars(seq, adjs) for seq, adjs in validation_datasets)

    def external_validation_datagen():
        yield from ()

    start_time = time.perf_counter()
    net, criterion, optimizer, writer = init_network()
    result = main(net, criterion, optimizer, writer, training_datagen, internal_validation_datagen,
                  external_validation_datagen, args.num_aa_to_process)
    time_elapsed = time.perf_counter() - start_time

    print(json.dumps(result))
    # profile.dump_stats('train_nn.lprof')
