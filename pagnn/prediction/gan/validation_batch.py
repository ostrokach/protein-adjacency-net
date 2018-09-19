import torch  # isort:skip
import concurrent.futures
import math
import os
import pickle
import tempfile
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from kmtools import sequence_tools

import pagnn
from pagnn import settings
from pagnn.prediction.gan import Args
from pagnn.training.gan import Args as ArgsTraining
from pagnn.training.gan import generate_noise
from pagnn.types import DataSetGAN

settings.CUDA = False


def main():
    args = Args.from_cli()

    # Training arguments
    args_training = ArgsTraining(root_path=args.work_path.parent)
    args_training.unique_name = args.work_path.name

    # Load network
    net_d = pagnn.models.AESeqAdjApplyExtra(
        'discriminator',
        hidden_size=args_training.hidden_size,
        bottleneck_size=1,
    )
    net_g = pagnn.models.AESeqAdjApplyExtra(
        'generator',
        hidden_size=args_training.hidden_size,
        bottleneck_size=16,
        encoder_network=net_d,
    )

    net_g.load_state_dict(
        torch.load(
            args_training.work_path.joinpath('models').joinpath(f'net_g-step_{args.step}.model')
            .as_posix(),
            map_location='cpu'))
    net_d.load_state_dict(
        torch.load(
            args_training.work_path.joinpath('models').joinpath(f'net_d-step_{args.step}.model')
            .as_posix(),
            map_location='cpu'))

    # Load input data
    if args.input_file.suffix == '.pdb':
        raise NotImplementedError("PDB support coming soon...")
    elif args.input_file.suffix == '.parquet':
        dataset = read_parquet_input(args)
    else:
        raise NotImplementedError("Only PDB and Parquet inputs are supported!")

    # Generate sequences
    random_state = np.random.RandomState(int(os.getenv('SLURM_ARRAY_TASK_ID', '42')))
    with args.validation_dataset_file.open('rb') as fin:
        datagen_ref = pickle.load(fin)
    df = generate_examples(args, dataset, datagen_ref, net_d, net_g, random_state)

    # Add secondary structure
    if args.include_ss:
        add_secondary_structure(args, df, copy=False)

    # Write output
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        args.output_file,
        version='2.0',
        flavor='spark',
    )


def read_parquet_input(args):
    rowgen = pagnn.io.iter_datarows(args.input_file)
    rowgen = list(rowgen)
    assert len(rowgen) == 1
    row = rowgen[0]
    dataset = pagnn.dataset.row_to_dataset(row, target=1)
    dataset = pagnn.dataset.dataset_to_gan(dataset)
    return dataset


def get_bottleneck_indices(adjs):
    idxs = []
    start = 0
    for i, adj in enumerate(adjs):
        stop = start + adj[4].shape[1]
        idxs.append((
            math.floor(start / 4),
            math.ceil(stop / 4),
        ))
        start = stop
    assert idxs[-1][1] == math.ceil(sum(adj[4].shape[1] for adj in adjs) / 4)
    return idxs


def generate_examples(args: Args, dataset: DataSetGAN, datagen_ref: DataSetGAN, net_d, net_g,
                      random_state) -> pd.DataFrame:
    datavar = net_d.dataset_to_datavar(dataset)

    # TODO: Eventually, eval mode should work better
    net_g.train()
    net_d.train()

    # DataFrame columns
    df_columns = OrderedDict([
        ('discriminator_score', [np.nan]),
        ('blosum62_score', [np.nan]),
        ('edit_score', [np.nan]),
        ('sequence', [dataset.seqs[0].decode()]),
        ('sequence_type', ['ref']),
    ])

    # Randomly select a reference dataset for each batch
    ref_idxs = random_state.choice(np.arange(len(datagen_ref)), args.batch_size - 1, replace=False)
    dataset_ref = [datagen_ref[i] for i in ref_idxs]
    datavar_ref = [net_d.dataset_to_datavar(ds) for ds in dataset_ref]
    seqs_ref = [dv.seqs[0:1, :, :] for dv in datavar_ref]
    adjs_ref = [dv.adjs for dv in datavar_ref]

    # Create a batch
    seqs_batch = torch.cat(
        seqs_ref[:args.batch_size // 2] + [datavar.seqs] + seqs_ref[args.batch_size // 2:], 2)
    adjs_batch = adjs_ref[:args.batch_size // 2] + [datavar.adjs] + adjs_ref[args.batch_size // 2:]

    bottleneck_idxs = get_bottleneck_indices(adjs_batch)
    assert len(bottleneck_idxs) == args.batch_size == len(adjs_batch)

    noise = generate_noise(net_g, adjs_batch)

    for _ in tqdm.tqdm(range(args.nseqs), total=args.nseqs, disable=not settings.SHOW_PROGRESSBAR):
        noisev = noise.normal_(0, 1)
        with torch.no_grad():
            fake_seq = net_g(noisev, adjs_batch).data
            assert fake_seq.shape[2] == seqs_batch.shape[2]
            fake_pred = net_d(fake_seq, adjs_batch)
            assert bottleneck_idxs[-1][1] <= fake_pred.shape[2] <= (bottleneck_idxs[-1][1] + 1)

        start = 0
        for i, adj in enumerate(adjs_batch):
            stop = start + adj[0].shape[1]
            if i == args.batch_size // 2:
                fake_seq_slice = fake_seq[:, :, start:stop]
            start = stop
        assert stop == fake_seq.shape[2]

        for i, (start, stop) in enumerate(bottleneck_idxs):
            if i == args.batch_size // 2:
                fake_pred_slice = fake_pred[:, :, start:stop]
        assert stop <= fake_pred.shape[2] <= (stop + 1)

        fake_seq_slice_onehot = pagnn.utils.argmax_onehot(fake_seq_slice)
        assert (fake_seq_slice_onehot.sum(dim=1) == 1).all()

        df_columns['discriminator_score'].append(float(fake_pred_slice.sigmoid().mean()))
        df_columns['blosum62_score'].append(
            float(pagnn.utils.score_blosum62(datavar.seqs[0, :, :].data, fake_seq_slice_onehot)))
        df_columns['edit_score'].append(
            float(pagnn.utils.score_edit(datavar.seqs[0, :, :].data, fake_seq_slice_onehot)))
        df_columns['sequence'].append(''.join(
            pagnn.AMINO_ACIDS[int(i)] for i in np.argmax(pagnn.to_numpy(fake_seq_slice), 1)[0]))
        df_columns['sequence_type'].append('gen')

    df = pd.DataFrame(df_columns, index=range(len(df_columns['sequence'])))
    assert (df['sequence'].str.len() == len(df['sequence'][0])).all()
    return df


def run_psipred(idx, sequence, hhblits_database, tmp_dir):
    fasta_file = Path(tmp_dir).joinpath(f'seq_{idx}.fasta')
    with fasta_file.open('wt') as fout:
        fout.write(f'> seq_{idx}\n')
        fout.write(sequence + '\n')
    psipred_file = sequence_tools.run_psipred(fasta_file, hhblits_database)
    psipred = sequence_tools.read_psipred(psipred_file)
    return psipred


def add_secondary_structure(args, df: pd.DataFrame, copy=True) -> pd.DataFrame:
    df = df.copy() if copy else df

    with concurrent.futures.ProcessPoolExecutor(args.nprocs) as p, \
            tempfile.TemporaryDirectory() as tmp_dir:
        df['sequence_ss'] = list(
            p.map(run_psipred, df.index, df['sequence'].values, repeat(args.hhblits_database),
                  repeat(tmp_dir)))

    return df


if __name__ == '__main__':
    main()
