import argparse
import hashlib
import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from PIL import Image
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
from pagnn.utils import (add_image, argmax_onehot, array_to_seq, make_weblogo, score_blosum62,
                         score_edit, to_numpy, to_tensor)

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
    work_path.joinpath('models').mkdir(exist_ok=True)
    work_path.joinpath('checkpoints').mkdir(exist_ok=True)

    # Set up network
    seq_length = 512
    nc = 20
    nz = 100

    net_d = DiscriminatorNet()
    net_g = GeneratorNet()

    input = torch.FloatTensor(args.batch_size, nc, seq_length)
    noise = torch.FloatTensor(args.batch_size, nz, 1)
    fixed_noise = torch.FloatTensor(args.batch_size, nz, 1).normal_(0, 1)
    loss = nn.BCELoss()
    one = torch.FloatTensor([1])
    mone = one * -1

    if settings.CUDA:
        net_g.cuda()
        net_d.cuda()
        input = input.cuda()
        loss = loss.cuda()
        one = one.cuda()
        mone = mone.cuda()
        noise = noise.cuda()
        fixed_noise = fixed_noise.cuda()

    if args.rmsprop:
        optimizer_d = optim.RMSprop(net_d.parameters(), lr=args.learning_rate_d)
        optimizer_g = optim.RMSprop(net_g.parameters(), lr=args.learning_rate_g)
    else:
        # Encouraged
        optimizer_d = optim.Adam(
            net_d.parameters(), lr=args.learning_rate_d, betas=(args.beta1, args.beta2))
        optimizer_g = optim.Adam(
            net_g.parameters(), lr=args.learning_rate_g, betas=(args.beta1, args.beta2))

    if args.resume:
        net_g.load_state_dict(
            torch.load(
                work_path.joinpath('models').joinpath(checkpoint['net_g_path_name']).as_posix()))
        net_d.load_state_dict(
            torch.load(
                work_path.joinpath('models').joinpath(checkpoint['net_d_path_name']).as_posix()))

    step = checkpoint.get('step', 0)
    stat = Stats()

    for p in net_d.parameters():
        p.requires_grad = False

    progressbar = tqdm.tqdm(disable=not settings.SHOW_PROGRESSBAR)
    while True:
        # === Train discriminator ===
        for m in net_d.modules():
            if isinstance(m, nn.Conv1d) and m.kernel_size == (2, ):
                for p in m.parameters():
                    p.requires_grad = True

        for _ in range(args.d_iters):
            # # Clamp parameters to a cube
            # for p in net_d.parameters():
            #     p.data.clamp_(args.clamp_lower, args.clamp_upper)

            net_d.zero_grad()

            # Pos
            pos_row = next(positive_rowgen)
            pos_ds = to_gan(row_to_dataset(pos_row, 1))
            pos_dv = dataset_to_datavar(pos_ds, 0)
            pos_pred = net_d(*pos_dv)
            pos_target = Variable(to_tensor(np.ones(1)).unsqueeze(1))
            pos_loss = loss(pos_pred, pos_target)
            pos_loss.backward(one)
            # pos_loss = torch.mean(pos_pred)
            # pos_loss.backward(mone * 2)

            # Neg
            neg_ds = negative_ds_gen.send(pos_ds)
            neg_dv = dataset_to_datavar(neg_ds, 0)
            neg_pred = net_d(*neg_dv)
            neg_target = Variable(to_tensor(np.zeros(args.batch_size)).unsqueeze(1))
            neg_loss = loss(neg_pred, neg_target)
            neg_loss.backward()
            # neg_loss = torch.mean(neg_pred)
            # neg_loss.backward(one)

            # gradient_penalty = calc_gradient_penalty(
            #     args, net_d, pos_dv.seqs.data, neg_dv.seqs.data,
            #     [adj.data for adj in pos_dv.adjs])
            # gradient_penalty.backward()

            optimizer_d.step()

            # Write stats
            stat.training_preds.extend([to_numpy(pos_pred).squeeze(), to_numpy(neg_pred).squeeze()])
            stat.training_targets.extend([np.ones(1), np.zeros(args.batch_size)])
            stat.pos_losses.append(to_numpy(pos_loss))
            stat.neg_losses.append(to_numpy(neg_loss))

        for p in net_d.parameters():
            p.requires_grad = True

        for m in net_d.modules():
            if isinstance(m, nn.Conv1d) and m.kernel_size == (2, ):
                for p in m.parameters():
                    p.requires_grad = False

        for _ in range(args.d_iters):
            # # Clamp parameters to a cube
            # for p in net_d.parameters():
            #     p.data.clamp_(args.clamp_lower, args.clamp_upper)

            net_d.zero_grad()

            # Real
            real_row = next(positive_rowgen)
            real_ds = to_gan(row_to_dataset(real_row, 1))
            real_dv = dataset_to_datavar(real_ds, 0)
            real_pred = net_d(*real_dv)
            real_target = Variable(to_tensor(np.ones(1)).unsqueeze(1))
            real_loss = loss(real_pred, real_target)
            real_loss.backward(one * 2)

            # Neg
            neg_ds = negative_ds_gen.send(real_ds)
            neg_dv = dataset_to_datavar(neg_ds, 0)
            neg_pred = net_d(*neg_dv)
            neg_target = Variable(to_tensor(np.zeros(args.batch_size)).unsqueeze(1))
            neg_loss = loss(neg_pred, neg_target)
            neg_loss.backward()

            # Fake
            noise.resize_(args.batch_size, nz, 1).normal_(0, 1)
            noisev = Variable(noise, volatile=True)  # totally freeze net_g
            fake = Variable(net_g(noisev, real_dv.adjs, net_d).data)
            fake_pred = net_d(fake, real_dv.adjs)
            fake_target = Variable(to_tensor(np.zeros(args.batch_size)).unsqueeze(1))
            fake_loss = loss(fake_pred, fake_target)
            fake_loss.backward()

            # Write stats
            stat.real_losses.append(to_numpy(real_loss))
            stat.fake_losses.append(to_numpy(fake_loss))

            optimizer_d.step()

        # === Train generator ===
        for p in net_d.parameters():
            p.requires_grad = False  # to avoid computation

        net_g.zero_grad()

        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(args.batch_size, nz, 1).normal_(0, 1)
        noisev = Variable(noise)
        g_fake = net_g(noisev, pos_dv.adjs, net_d)
        d_fake_pred = net_d(g_fake, pos_dv.adjs)
        d_fake_target = Variable(to_tensor(np.ones(args.batch_size)).unsqueeze(1))
        g_fake_loss = loss(d_fake_pred, d_fake_target)
        g_fake_loss.backward()
        # g_fake_loss = torch.mean(d_fake_pred)
        # g_fake_loss.backward(mone)

        stat.g_fake_losses.append(to_numpy(g_fake_loss))

        optimizer_g.step()

        # === Write ===
        if step % args.steps_between_checkpoins == 0:
            scores = calculate_statistics_basic(args, stat)

            resume_checkpoint = args.resume and step == checkpoint.get('step')

            if resume_checkpoint or (step % args.steps_between_extended_checkpoins == 0):
                scores_extended = calculate_statistics_extended(args, stat, net_d, net_g,
                                                                internal_validation_datasets,
                                                                external_validation_datasets)
                scores.update(scores_extended)

            if resume_checkpoint:
                validate_checkpoint(checkpoint, scores)
            else:
                write_checkpoint(step, stat, scores, work_path, writer, net_d, net_g,
                                 current_performance)

        step += 1
        stat = Stats()
        progressbar.update()


def calc_gradient_penalty(args, net_d, real_data, fake_data, adjs, lambda_=10):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(args.batch_size,
                         fake_data.nelement() // args.batch_size).contiguous().view(
                             args.batch_size, 20, 512)
    if settings.CUDA:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if settings.CUDA:
        interpolates = interpolates.cuda()

    interpolates = autograd.Variable(interpolates, requires_grad=True)
    adjs = [Variable(adj) for adj in adjs]

    # TODO: Don't forget that we changed this
    interpolates = F.softmax(interpolates, 1)

    disc_interpolates = net_d(interpolates, adjs)

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


def calculate_statistics_basic(args: argparse.Namespace, stat: Stats, _prev_stats={}):
    scores = {}

    # Training accuracy
    if stat.training_preds and stat.training_targets:
        training_preds_ar = np.hstack(stat.training_preds)
        training_targets_ar = np.hstack(stat.training_targets)
        scores['training_auc'] = metrics.roc_auc_score(training_targets_ar, training_preds_ar)

    # Runtime
    prev_validation_time = _prev_stats.get('validation_time')
    _prev_stats['validation_time'] = time.perf_counter()
    if prev_validation_time:
        scores['time_between_checkpoins'] = time.perf_counter() - prev_validation_time
        scores['checkpoins_per_second'] = (args.steps_between_checkpoins /
                                           (_prev_stats['validation_time'] - prev_validation_time))

    return scores


def calculate_statistics_extended(args: argparse.Namespace, stat: Stats, net_d, net_g,
                                  internal_validation_datasets, external_validation_datasets):
    stat.is_extended = True

    scores = {}
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

    # BLOSUM 62 and edit distance
    noise = torch.FloatTensor(128, 100, 1)
    if settings.CUDA:
        noise = noise.cuda()

    name = 'validation_gan_exact_80_1000'
    for i, dataset in enumerate(internal_validation_datasets[name]):
        seq_wt = dataset.seqs[0].decode()
        start = 0
        stop = start + len(seq_wt)
        datavar = dataset_to_datavar(dataset, volatile=True, offset=start)
        noisev = Variable(noise.normal_(0, 1), volatile=True)
        pred = net_g(noisev, datavar.adjs, net_d)
        pred_argmax = argmax_onehot(pred[:, :, start:stop].data)
        target = datavar.seqs[0, :, start:stop].data
        stat.blosum62_scores.append(score_blosum62(target, pred_argmax))
        stat.edit_scores.append(score_edit(target, pred_argmax))

        # WebLogo image
        if i == 0:
            designed_seqs = [
                array_to_seq(to_numpy(pred[i, :, start:stop])) for i in range(pred.shape[0])
            ]
            weblogo_wt = make_weblogo([seq_wt], units='probability', color_scheme='chemistry')
            weblogo_design = make_weblogo(
                designed_seqs, units='probability', color_scheme='chemistry')
            if weblogo_wt is not None and weblogo_design is not None:
                stat.weblogo1 = Image.fromarray(np.vstack([weblogo_design, weblogo_wt]))

    scores[name + '-blosum62'] = np.mean(stat.blosum62_scores)
    scores[name + '-edit'] = np.mean(stat.edit_scores)

    return scores


def get_designed_seqs(args, dataset, net_d, net_g):
    seq_wt = dataset.seqs[0].decode()

    designed_seqs = []
    noise = torch.FloatTensor(args.batch_size, 100, 1)
    if settings.CUDA:
        noise = noise.cuda()
    # for offset in range(0, max(1, 512 - len(seq_wt))):
    for offset in range(0, 1):
        start = offset
        stop = start + len(seq_wt)
        datavar = dataset_to_datavar(dataset, volatile=True, offset=offset)
        noisev = Variable(noise.normal_(0, 1))
        pred = net_g(noisev, datavar.adjs, net_d)
        pred_np = pagnn.to_numpy(pred)
        pred_np = pred_np[:, :, start:stop]
        designed_seqs.extend([array_to_seq(pred_np[i]) for i in range(pred_np.shape[0])])
    return designed_seqs


def validate_checkpoint(checkpoint, scores):
    common_scores = set(checkpoint) & set(scores)
    assert common_scores
    assert all(checkpoint[s] == scores[s] for s in common_scores)


def write_checkpoint(step, stat, scores, work_path, writer, net_d, net_g, current_performance):
    # Network parameters
    # for name, param in net_g.named_parameters():
    #     writer.add_histogram("net_g_" + name, to_numpy(param), step)
    # for name, param in net_d.named_parameters():
    #     writer.add_histogram("net_d_" + name, to_numpy(param), step)

    checkpoint = {
        'step': step,
        'unique_name': work_path.name,
        # **scores,
    }

    # Scalars
    for score_name, score_value in scores.items():
        if score_value is None:
            logger.warning(f"Score {score_name} is None!")
            continue
        writer.add_scalar(score_name, score_value, step)

    writer.add_scalar('pos_loss', np.hstack(stat.pos_losses).mean(), step)
    writer.add_scalar('neg_loss', np.hstack(stat.neg_losses).mean(), step)
    # writer.add_scalar('real_loss', np.hstack(stat.real_losses).mean(), step)
    writer.add_scalar('fake_loss', np.hstack(stat.fake_losses).mean(), step)
    writer.add_scalar('g_fake_loss', np.hstack(stat.g_fake_losses).mean(), step)

    # Histograms
    writer.add_histogram('pos_losses', np.hstack(stat.pos_losses), step)
    writer.add_histogram('neg_losses', np.hstack(stat.neg_losses), step)
    # writer.add_histogram('real_losses', np.hstack(stat.real_losses), step)
    writer.add_histogram('fake_losses', np.hstack(stat.fake_losses), step)
    writer.add_histogram('g_fake_losses', np.hstack(stat.g_fake_losses), step)
    writer.add_histogram('blosum62_scores', np.array(stat.blosum62_scores), step)
    writer.add_histogram('edit_scores', np.array(stat.edit_scores), step)

    # PR curves
    writer.add_pr_curve('training', np.hstack(stat.training_targets), np.hstack(
        stat.training_preds), step)

    # Images
    if stat.weblogo1 is not None:
        add_image(writer, 'weblogo1', stat.weblogo1, step)

    if stat.weblogo1 is not None:
        # Save model
        net_d_dump_path = work_path.joinpath('models').joinpath(f'net_d-step_{step}.model')
        torch.save(net_d.state_dict(), net_d_dump_path.as_posix())
        checkpoint['net_d_path_name'] = net_d_dump_path.name

        net_g_dump_path = work_path.joinpath('models').joinpath(f'net_g-step_{step}.model')
        torch.save(net_g.state_dict(), net_g_dump_path.as_posix())
        checkpoint['net_g_path_name'] = net_g_dump_path.name

        # Save checkpoint
        with work_path.joinpath('checkpoints').joinpath(f'checkpoint-step{step}.json').open(
                'wt') as fout:
            json.dump(checkpoint, fout, sort_keys=True, indent=4)
        with work_path.joinpath('checkpoints').joinpath('checkpoint.json').open('wt') as fout:
            json.dump(checkpoint, fout, sort_keys=True, indent=4)

    if current_performance is not None:
        current_performance.update(writer.scalar_dict)


def get_log_dir(args) -> str:
    args_dict = vars(args)
    state_keys = ['rmsprop', 'learning_rate_d', 'learning_rate_g', 'weight_decay', 'n_filters']
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
    # for mutation_class in ['protherm', 'humsavar']:
    for mutation_class in ['protherm']:
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

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    args.steps_between_checkpoins = args.steps_between_checkpoins
    args.steps_between_extended_checkpoins = args.steps_between_extended_checkpoins

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
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # # Add additional functions to profile
    # lp.add_function(score_edit)
    # lp.add_function(score_blosum62)
    # lp.add_function(calculate_statistics_basic)
    # lp.add_function(calculate_statistics_extended)
    # lp.add_function(evaluate_validation_dataset)
    # lp.add_function(evaluate_mutation_dataset)
    # lp.add_function(train)
    # # Profile the main function
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # # Print results
    # lp.print_stats()
