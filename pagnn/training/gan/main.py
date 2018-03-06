import argparse
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Mapping, Optional, Tuple, Union

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
from pagnn.dataset import row_to_dataset, to_gan
from pagnn.datavargan import dataset_to_datavar
from pagnn.models import DiscriminatorNet, GeneratorNet
from pagnn.training.common import get_rowgen_neg, get_rowgen_pos
from pagnn.training.gan import (basic_permuted_sequence_adder, get_mutation_dataset,
                                get_validation_dataset, parse_args)
from pagnn.types import DataRow, DataSetGAN

logger = logging.getLogger(__name__)

get_datagen_gan = None
training_datagen = None


def train(args: argparse.Namespace,
          work_path: Path,
          writer: SummaryWriter,
          positive_rowgen,
          negative_ds_gen,
          internal_validation_datasets,
          external_validation_datasets,
          current_performance: Optional[Dict[str, Union[str, float]]] = None):
    """"""
    seq_length = 512
    nc = 20
    nz = 100

    net_g = GeneratorNet()
    net_d = DiscriminatorNet()

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
        optimizer_d = optim.Adam(
            net_d.parameters(), lr=args.learning_rate_d, betas=(args.beta1, 0.999))
        optimizer_g = optim.Adam(
            net_g.parameters(), lr=args.learning_rate_g, betas=(args.beta1, 0.999))
    else:
        # Encouraged
        optimizer_d = optim.RMSprop(net_d.parameters(), lr=args.learning_rate_d)
        optimizer_g = optim.RMSprop(net_g.parameters(), lr=args.learning_rate_g)

    progressbar = tqdm.tqdm(disable=not settings.SHOW_PROGRESSBAR)
    while True:
        # === Train discriminator ===
        net_d.zero_grad()

        for p in net_d.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in net_d update

        # Clamp parameters to a cube
        for p in net_d.parameters():
            p.data.clamp_(args.clamp_lower, args.clamp_upper)

        for _ in range(args.d_iters):
            pos_row = next(positive_rowgen)
            pos_ds = to_gan(row_to_dataset(pos_row, 1))
            pos_dv = dataset_to_datavar(pos_ds)
            loss = net_d(*pos_dv)
            loss.backward(one * args.batch_size)

            neg_ds = negative_ds_gen.send(pos_ds)
            neg_dv = dataset_to_datavar(neg_ds)
            loss = net_d(*neg_dv)
            loss.backward(mone)

        pos_row = next(positive_rowgen)
        pos_ds = to_gan(row_to_dataset(pos_row, 1))
        pos_dv = dataset_to_datavar(pos_ds)
        loss_real = net_d(*pos_dv)
        loss_real.backward(one * args.batch_size)

        noise.resize_(args.batch_size, nz, 1).normal_(0, 1)
        noisev = Variable(noise, volatile=True)  # totally freeze net_g
        fake = Variable(net_g(noisev, pos_dv.adjs).data)
        loss_fake = net_d(fake, pos_dv.adjs)
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
        fake = net_g(noisev, pos_dv.adjs)
        loss_fake = net_d(fake, pos_dv.adjs)
        loss_fake.backward(one)
        optimizer_g.step()

        error_d = loss_real - loss_fake

        progressbar.update()
        print(error_g, error_d)


def get_log_dir(args) -> str:
    args_dict = vars(args)
    state_keys = ['adam', 'learning_rate_d', 'learning_rate_g', 'weight_decay', 'n_filters']
    state_dict = {k: args_dict[k] for k in state_keys}
    # https://stackoverflow.com/a/22003440/2063031
    state_hash = hashlib.md5(json.dumps(state_dict, sort_keys=True).encode('ascii')).hexdigest()
    log_dir = '-'.join([
        Path(__file__).stem, pagnn.__version__, args.training_methods, args.training_permutations,
        str(args.training_min_seq_identity), state_hash
    ] + ([args.tag] if args.tag else []))
    return log_dir


def get_training_datasets(args: argparse.Namespace, root_path: Path, data_path: Path
                         ) -> Tuple[Iterator[DataRow], Generator[DataSetGAN, DataSetGAN, None]]:
    logger.info("Setting up training datagen...")
    positive_rowgen = get_rowgen_pos(
        'training',
        args.training_min_seq_identity,
        data_path,
        random_state=None,
    )
    negative_rowgen = get_rowgen_neg(
        'training',
        args.training_min_seq_identity,
        data_path,
        random_state=None,
    )
    del negative_rowgen
    if '.' not in args.training_methods:
        negative_ds_gen = basic_permuted_sequence_adder(
            num_sequences=args.batch_size,
            keep_pos=False,
            random_state=None,
        )
    else:
        raise NotImplementedError()
    next(negative_ds_gen)
    return positive_rowgen, negative_ds_gen


def get_internal_validation_datasets(args, root_path, data_path) -> Mapping[str, List[DataSetGAN]]:
    logger.info("Setting up validation datagen...")

    internal_validation_datasets: Dict[str, List[DataSetGAN]] = {}
    for method in args.validation_methods.split('.'):
        datagen_name = (f'validation_gan_{method}_{args.validation_min_seq_identity}'
                        f'_{args.validation_num_sequences}')
        cache_file = root_path.joinpath(datagen_name + '.pickle')
        try:
            with cache_file.open('rb') as fin:
                dataset = pickle.load(fin)
            assert len(dataset) == args.validation_num_sequences
            logger.info("Loaded validation datagen from file: '%s'.", cache_file)
        except FileNotFoundError:
            logger.info("Generating validation datagen: '%s'.", datagen_name)
            random_state = np.random.RandomState(sum(ord(c) for c in method))
            dataset = get_validation_dataset(args, method, data_path, random_state)

            with cache_file.open('wb') as fout:
                pickle.dump(dataset, fout, pickle.HIGHEST_PROTOCOL)

        internal_validation_datasets[datagen_name] = dataset

    return internal_validation_datasets


def get_external_validation_datasets(args, root_path, data_path) -> Mapping[str, List[DataSetGAN]]:
    external_validation_datagens: Dict[str, List[DataSetGAN]] = {}
    for mutation_class in ['protherm', 'humsavar']:
        external_validation_datagens[f'validation_{mutation_class}'] = get_mutation_dataset(
            mutation_class, data_path)
    return external_validation_datagens


def main():
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
    # cache_path = data_path

    unique_name = get_log_dir(args)
    work_path = root_path.joinpath(unique_name)
    tensorboard_path = work_path.joinpath('tensorboard')
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    # === Training ===
    positive_rowgen, negative_ds_gen = get_training_datasets(args, root_path, data_path)

    # === Internal Validation ===
    internal_validation_datasets = get_internal_validation_datasets(args, root_path, data_path)

    # === Mutation Validation ===
    external_validation_datasets = get_external_validation_datasets(args, root_path, data_path)

    # === Train ===
    start_time = time.perf_counter()
    result: Dict[str, Union[str, float]] = {}
    writer = SummaryWriter(tensorboard_path.as_posix())
    try:
        train(
            args,
            work_path,
            writer,
            positive_rowgen,
            negative_ds_gen,
            internal_validation_datasets,
            external_validation_datasets,
            current_performance=result)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
    result['time_elapsed'] = time.perf_counter() - start_time

    # === Output ===
    print(json.dumps(result, sort_keys=True, indent=4))


if __name__ == '__main__':
    # === Basic ===
    main()
    # === Profiled ===
    # from pagnn.datavargan import push_adjs, push_seqs
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # # Add additional functions to profile
    # lp.add_function(push_adjs)
    # lp.add_function(push_seqs)
    # lp.add_function(dataset_to_datavar)
    # lp.add_function(train)
    # # Profile the main function
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # # Print results
    # lp.print_stats()
