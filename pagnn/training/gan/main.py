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
from scipy import stats
# from scipy import stats
# from memory_profiler import profile
# from line_profiler import LineProfiler
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch import autograd
from torch.autograd import Variable

import pagnn
from pagnn import settings
from pagnn.dataset import row_to_dataset, to_gan
from pagnn.datavargan import dataset_to_datavar
from pagnn.models import DiscriminatorNet, GeneratorNet
from pagnn.training.common import get_rowgen_neg, get_rowgen_pos
from pagnn.training.gan import (Stats, basic_permuted_sequence_adder, evaluate_mutation_dataset,
                                evaluate_validation_dataset, get_mutation_dataset,
                                get_validation_dataset, parse_args)
from pagnn.types import DataRow, DataSetGAN
from pagnn.utils import to_numpy

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
          checkpoint,
          current_performance: Optional[Dict[str, Union[str, float]]] = None):
    """"""
    models_path = work_path.joinpath('models')
    models_path.mkdir(exist_ok=True)

    # Set up network
    seq_length = 512
    nc = 20
    nz = 100

    net_g = GeneratorNet()
    net_d = DiscriminatorNet()

    input = torch.FloatTensor(args.batch_size, nc, seq_length)
    noise = torch.FloatTensor(args.batch_size, nz, 1)
    fixed_noise = torch.FloatTensor(args.batch_size, nz, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
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

    if args.resume:
        net_g.load_state_dict(
            torch.load(models_path.joinpath(checkpoint['net_g_path_name']).as_posix()))
        net_d.load_state_dict(
            torch.load(models_path.joinpath(checkpoint['net_d_path_name']).as_posix()))

    step = checkpoint.get('step', 0)
    stat = Stats()

    progressbar = tqdm.tqdm(disable=not settings.SHOW_PROGRESSBAR)
    while True:
        # === Train discriminator ===
        net_d.zero_grad()

        for p in net_d.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in net_d update

        for _ in range(args.d_iters):
            pos_row = next(positive_rowgen)
            pos_ds = to_gan(row_to_dataset(pos_row, 1))
            pos_dv = dataset_to_datavar(pos_ds)
            pos_pred = net_d(*pos_dv)
            pos_loss = torch.mean(pos_pred)
            pos_loss.backward(mone)

            neg_ds = negative_ds_gen.send(pos_ds)
            neg_dv = dataset_to_datavar(neg_ds)
            neg_pred = net_d(*neg_dv)
            neg_loss = torch.mean(neg_pred)
            neg_loss.backward(one)

            stat.training_preds.extend([to_numpy(pos_pred).squeeze(), to_numpy(neg_pred).squeeze()])
            stat.training_targets.extend([np.ones(1), np.zeros(args.batch_size)])
            stat.pos_losses.append(to_numpy(pos_loss))
            stat.neg_losses.append(to_numpy(neg_loss))

        pos_row = next(positive_rowgen)
        pos_ds = to_gan(row_to_dataset(pos_row, 1))
        pos_dv = dataset_to_datavar(pos_ds)
        real_pred = net_d(*pos_dv)
        real_loss = torch.mean(real_pred)
        real_loss.backward(mone)

        noise.resize_(args.batch_size, nz, 1).normal_(0, 1)
        noisev = Variable(noise, volatile=True)  # totally freeze net_g
        fake = Variable(net_g(noisev, pos_dv.adjs).data)
        fake_pred = net_d(fake, pos_dv.adjs)
        fake_loss = torch.mean(fake_pred)
        fake_loss.backward(one)

        optimizer_d.step()

        stat.error_g = to_numpy(real_loss - fake_loss)
        stat.real_losses.append(to_numpy(real_loss))
        stat.fake_losses.append(to_numpy(fake_loss))

        # Clamp parameters to a cube
        for p in net_d.parameters():
            p.data.clamp_(args.clamp_lower, args.clamp_upper)

        # === Train discriminator ===
        for p in net_d.parameters():
            p.requires_grad = False  # to avoid computation

        net_g.zero_grad()

        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(args.batch_size, nz, 1).normal_(0, 1)
        noisev = Variable(noise)
        g_fake = net_g(noisev, pos_dv.adjs)
        d_fake_pred = net_d(g_fake, pos_dv.adjs)
        g_fake_loss = torch.mean(d_fake_pred)
        g_fake_loss.backward(mone)
        optimizer_g.step()

        stat.error_d = to_numpy(real_loss - g_fake_loss)
        stat.g_fake_losses.append(to_numpy(g_fake_loss))

        # === Write ===
        if step % args.steps_between_validation == 0:
            scores = calculate_statistics(args, stat, net_d, internal_validation_datasets,
                                          external_validation_datasets)
            if args.resume and step == checkpoint.get('step'):
                validate_checkpoint(checkpoint, scores)
            else:
                write_checkpoint(step, stat, scores, work_path, models_path, writer, net_d, net_g,
                                 current_performance)

        step += 1
        stat = Stats()
        progressbar.update()


def calc_gradient_penalty(args, net_d, real_data, fake_data, lambda_):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(args.batch_size,
                         real_data.nelement() / args.batch_size).contiguous().view(
                             args.batch_size, 3, 32, 32)
    if settings.CUDA:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if settings.CUDA:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = net_d(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda()
        if settings.CUDA else torch.ones(disc_interpolates.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * lambda_
    return gradient_penalty


def calculate_statistics(args: argparse.Namespace,
                         stat: Stats,
                         net_d,
                         internal_validation_datasets,
                         external_validation_datasets,
                         _prev_stats={}):
    scores = {}

    # Training accuracy
    if stat.training_preds and stat.training_targets:
        training_preds_ar = np.hstack(stat.training_preds)
        training_targets_ar = np.hstack(stat.training_targets)
        scores['training_auc'] = metrics.roc_auc_score(training_targets_ar, training_preds_ar)
    scores['error_g'] = stat.error_g
    scores['error_d'] = stat.error_d

    # Validation accuracy
    for name, datasets in internal_validation_datasets.items():
        targets_valid, outputs_valid = evaluate_validation_dataset(net_d, datasets)
        scores[name] = metrics.roc_auc_score(targets_valid, outputs_valid)

    for name, datasets in external_validation_datasets.items():
        targets_valid, outputs_valid = evaluate_mutation_dataset(net_d, datasets)
        if 'protherm' in name:
            # Protherm predicts ΔΔG, so positive values are destabilizing
            scores[name + '-spearman_r'] = stats.spearmanr(-targets_valid,
                                                           outputs_valid).correlation
        elif 'humsavar' in name:
            # For humsavar: 0 = stable, 1 = deleterious
            scores[name + '-auc'] = metrics.roc_auc_score(1 - targets_valid, outputs_valid)
        else:
            scores[name] = metrics.roc_auc_score(targets_valid + 1, outputs_valid)

    # Runtime
    prev_validation_time = _prev_stats.get('validation_time')
    _prev_stats['validation_time'] = time.perf_counter()
    if prev_validation_time:
        scores['validation_time'] = time.perf_counter() - prev_validation_time
        scores['iterations_per_second'] = (args.steps_between_validation /
                                           (_prev_stats['validation_time'] - prev_validation_time))

    return scores


def validate_checkpoint(checkpoint, scores):
    common_scores = set(checkpoint) & set(scores)
    assert common_scores
    assert all(checkpoint[s] == scores[s] for s in common_scores)


def write_checkpoint(step, stat, scores, work_path, models_path, writer, net_d, net_g,
                     current_performance):
    # Network parameters
    # for name, param in net_g.named_parameters():
    #     writer.add_histogram("net_g_" + name, to_numpy(param), step)
    # for name, param in net_d.named_parameters():
    #     writer.add_histogram("net_d_" + name, to_numpy(param), step)

    # Scalars
    for score_name, score_value in scores.items():
        writer.add_scalar(score_name, score_value, step)

    # Histograms
    writer.add_histogram('pos_losses', np.hstack(stat.pos_losses), step)
    writer.add_histogram('neg_losses', np.hstack(stat.neg_losses), step)
    writer.add_histogram('real_losses', np.hstack(stat.real_losses), step)
    writer.add_histogram('fake_losses', np.hstack(stat.fake_losses), step)
    writer.add_histogram('g_fake_losses', np.hstack(stat.g_fake_losses), step)

    # PR curves
    writer.add_pr_curve('training', np.hstack(stat.training_targets), np.hstack(
        stat.training_preds), step)

    # Save model
    net_d_dump_path = work_path.joinpath('models').joinpath(f'net_d-step_{step}.model')
    torch.save(net_d.state_dict(), net_d_dump_path.as_posix())

    net_g_dump_path = work_path.joinpath('models').joinpath(f'net_g-step_{step}.model')
    torch.save(net_g.state_dict(), net_g_dump_path.as_posix())

    # Save checkpoint
    checkpoint = {
        'step': step,
        'unique_name': work_path.name,
        'net_d_path_name': net_d_dump_path.name,
        'net_g_path_name': net_g_dump_path.name,
        # **scores,
    }
    with work_path.joinpath(f'checkpoint-step{step}.json').open('wt') as fout:
        json.dump(checkpoint, fout, sort_keys=True, indent=4)
    with work_path.joinpath('checkpoint.json').open('wt') as fout:
        json.dump(checkpoint, fout, sort_keys=True, indent=4)

    if current_performance is not None:
        current_performance.update(writer.scalar_dict)


def get_log_dir(args) -> str:
    args_dict = vars(args)
    state_keys = ['adam', 'learning_rate_d', 'learning_rate_g', 'weight_decay', 'n_filters']
    state_dict = {k: args_dict[k] for k in state_keys}
    # https://stackoverflow.com/a/22003440/2063031
    state_hash = hashlib.md5(json.dumps(state_dict, sort_keys=True).encode('ascii')).hexdigest()
    log_dir = '-'.join([
        Path(__file__).parent.name, pagnn.__version__, args.training_methods,
        args.training_permutations,
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


def load_checkpoint(args, work_path):
    info_file = work_path.joinpath('info.json')
    checkpoint_file = work_path.joinpath('checkpoint.json')

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
    else:
        if info_file.is_file():
            raise Exception(f"Info file '{info_file}' already exists!")
        args_dict = vars(args)
        with info_file.open('wt') as fout:
            json.dump(args_dict, fout, sort_keys=True, indent=4)
    return checkpoint


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

    # === Checkpoint ===
    checkpoint = load_checkpoint(args, work_path)

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
            checkpoint,
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
