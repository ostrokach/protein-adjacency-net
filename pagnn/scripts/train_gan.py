import argparse
import hashlib
import itertools
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import numpy as np
import torch
# import torch.nn as nn
import torch.optim as optim
import tqdm
# from scipy import stats
# from memory_profiler import profile
# from line_profiler import LineProfiler
# from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import pagnn
from pagnn import settings
from pagnn.dataset import row_to_dataset
from pagnn.models import Discriminator, Generator
from pagnn.scripts._train_gan import dataset_to_datavar, negative_sequence_adder, to_gan
from pagnn.scripts.common import get_rowgen_neg, get_rowgen_pos
from pagnn.types import DataGen, DataSetCollection

logger = logging.getLogger(__name__)

get_datagen_gan = None
training_datagen = None


def main(args: argparse.Namespace,
         work_path: Path,
         writer: SummaryWriter,
         positive_datagen,
         negative_datagen,
         batch_size=256,
         steps_between_validation=25_600,
         current_performance: Optional[Dict[str, Union[str, float]]] = None):
    """"""
    seq_length = 512
    nc = 20
    nz = 100

    net_g = Generator()
    net_d = Discriminator()

    input = torch.FloatTensor(args.batch_size, nc, seq_length)
    noise = torch.FloatTensor(args.batch_size, nz, 1)
    fixed_noise = torch.FloatTensor(args.batch_size, nz, 1).normal_(0, 1)
    one = torch.FloatTensor([args.batch_size])
    mone = one * -1

    if settings.CUDA:
        net_g.cuda()
        net_d.cuda()
        input = input.cuda()
        one = one.cuda()
        mone = mone.cuda()
        noise = noise.cuda()
        fixed_noise = fixed_noise.cuda()

    if args.adam:
        optimizer_d = optim.Adam(net_d.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
        optimizer_g = optim.Adam(net_g.parameters(), lr=args.lrG, betas=(args.beta1, 0.999))
    else:
        # Encouraged
        optimizer_d = optim.RMSprop(net_d.parameters(), lr=args.lrD)
        optimizer_g = optim.RMSprop(net_g.parameters(), lr=args.lrG)

    while True:

        def next_dataset(rowgen):
            row = next(rowgen)
            dataset = row_to_dataset(row)
            dataset = to_gan(dataset)
            return dataset

        # === Train discriminator ===

        net_d.zero_grad()

        for p in net_d.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in net_d update

        # Clamp parameters to a cube
        for p in net_d.parameters():
            p.data.clamp_(args.clamp_lower, args.clamp_upper)

        for _ in range(args.num_train_d):
            pos_ds = next(positive_datagen)
            loss = net_d(dataset_to_datavar(pos_ds))
            loss.backward(one * args.batch_size)

            neg_ds = negative_datagen.send(pos_ds)
            loss = net_d(dataset_to_datavar(neg_ds))
            loss.backward(mone)

        pos_ds = next(positive_datagen)
        pos_dv = dataset_to_datavar(pos_ds)
        loss_real = net_d(pos_dv)
        loss_real.backward(one * args.batch_size)

        noise.resize_(args.batch_size, nz, 1).normal_(0, 1)
        noisev = Variable(noise, volatile=True)  # totally freeze net_g
        fake = Variable(net_g((noisev, pos_dv.adjs)).data)
        loss_fake = net_d((fake, pos_dv.adjs))
        loss_fake.backward(mone)
        optimizer_d.step()

        error_g = loss_real - loss_fake

        # === Train discriminator ===
        for p in net_d.parameters():
            p.requires_grad = False  # to avoid computation

        net_g.zero_grad()

        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(args.batch_size, nz, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = net_g((noisev, pos_dv.adjs))
        loss_fake = net_d((fake, pos_dv.adjs))
        loss_fake.backward(one)
        optimizer_g.step()

        error_d = loss_real - loss_fake

        print(error_g, error_d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Paths
    # # Location where to create subfolders for storing network data and cache files.
    parser.add_argument('--rootdir', type=str, default='.')
    # # Location of the `adjacency-net` databin folder.
    parser.add_argument('--datadir', type=str, default='.')
    # Network parameters
    parser.add_argument('--loss_name', type=str, default='BCELoss')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--n_filters', type=int, default=64)
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64)
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
    # TODO(AS): Run concurrent jobs in the computer has multiple GPUs
    parser.add_argument('-n', '--num-concurrent-jobs', type=int, default=1)
    args = parser.parse_args()
    return args


def get_log_dir(args) -> str:
    args_dict = vars(args)
    state_keys = ['loss_name', 'lr', 'weight_decay', 'n_filters']
    state_dict = {k: args_dict[k] for k in state_keys}
    # https://stackoverflow.com/a/22003440/2063031
    state_hash = hashlib.md5(json.dumps(state_dict, sort_keys=True).encode('ascii')).hexdigest()
    log_dir = '-'.join([
        Path(__file__).stem, pagnn.__version__, args.training_methods, args.training_permutations,
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
    positive_rowgen = get_rowgen_pos(
        'training',
        args.training_min_seq_identity,
        data_path,
        random_state=None,
    )

    # === Internal Validation ===
    logger.info("Setting up validation datagen...")
    internal_validation_datagens: Dict[str, DataGen] = {}
    min_seq_identity = 80
    for method in args.validation_methods.split('.'):
        datagen_name = (f'validation_gan_{method}_{args.validation_min_seq_identity}'
                        f'_{args.validation_num_sequences}')
        cache_file = root_path.joinpath(datagen_name + '.pickle')
        try:
            with cache_file.open('rb') as fin:
                dataset = pickle.load(fin)
            logger.info("Loaded validation datagen from file: '%s'.", cache_file)
            assert len(dataset) == args.validation_num_sequences
        except FileNotFoundError:
            logger.info("Generating validation datagen: '%s'.", datagen_name)
            negative_rowgen = get_rowgen_neg(
                'validation',
                args.validation_min_seq_identity,
                data_path,
                random_state=np.random.RandomState(42),
            )
            nsa = negative_sequence_adder(
                negative_rowgen(),
                method,
                args.validation_num_sequences,
                keep_pos=True,
                random_state=np.random.RandomState(42))
            next(nsa)
            dataset = [
                nsa.send(to_gan(row_to_dataset(r, 1)))
                for r in tqdm.tqdm(
                    itertools.islice(negative_rowgen, args.validation_num_sequences),
                    total=args.validation_num_sequences,
                    desc=cache_file.name,
                    disable=not settings.SHOW_PROGRESSBAR)
            ]
            assert len(dataset) == args.validation_num_sequences
            with cache_file.open('wb') as fout:
                pickle.dump(dataset, fout, pickle.HIGHEST_PROTOCOL)

        def datagen_from_memory() -> Iterator[DataSetCollection]:
            return dataset

        internal_validation_datagens[datagen_name] = datagen_from_memory

    # === Mutation Validation ===
    external_validation_datagens: Dict[str, DataGen] = {}
    for mutation_class in ['protherm', 'humsavar']:
        external_validation_datagens[
            f'validation_{mutation_class}'] = pagnn.get_mutation_datagen_gan(
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
