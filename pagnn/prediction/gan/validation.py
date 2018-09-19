import torch  # isort:skip

import concurrent.futures
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
    df = generate_examples(args, dataset, net_d, net_g)

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


def generate_examples(args: Args, dataset: DataSetGAN, net_d, net_g) -> pd.DataFrame:
    datavar = net_d.dataset_to_datavar(dataset)
    seqs = datavar.seqs
    adjs = [datavar.adjs]
    min_length = adjs[0][4].shape[1]

    # TODO: Eventually, eval mode should work better
    net_g.train()
    net_d.train()

    target = torch.cat([seqs[0, :, :].data] * args.batch_size, 1)
    assert (target.sum(dim=0) == 1).all()

    noise = generate_noise(net_g, adjs * args.batch_size)

    # DataFrame columns
    df_columns = OrderedDict([
        ('total_discriminator_score', [np.nan]),
        ('total_blosum62_score', [np.nan]),
        ('total_edit_score', [np.nan]),
        ('best_discriminator_score', [np.nan]),
        ('best_blosum62_score', [np.nan]),
        ('best_edit_score', [np.nan]),
        ('sequence', [dataset.seqs[0].decode()]),
        ('sequence_type', ['ref']),
        ('best_idx', [-1]),
    ])

    for _ in tqdm.tqdm(range(args.nseqs), total=args.nseqs, disable=not settings.SHOW_PROGRESSBAR):
        noisev = noise.normal_(0, 1)
        with torch.no_grad():
            fake_seq = net_g(noisev, adjs * args.batch_size).data
            assert fake_seq.shape[2] == (seqs.shape[2] * args.batch_size)
            fake_pred = net_d(fake_seq, adjs * args.batch_size)
            # Not sure why, but seems to work this way...
            assert (min_length * args.batch_size / 4) <= fake_pred.shape[2] <= (
                min_length * args.batch_size / 4 + 1), (fake_pred.shape,
                                                        min_length * args.batch_size / 4 + 1)

        fake_seq_onehot = pagnn.utils.argmax_onehot(fake_seq)
        assert (fake_seq_onehot.sum(dim=1) == 1).all()

        df_columns['total_discriminator_score'].append(float(fake_pred.sigmoid().mean()))
        df_columns['total_blosum62_score'].append(
            float(pagnn.utils.score_blosum62(target, fake_seq_onehot)))
        df_columns['total_edit_score'].append(
            float(pagnn.utils.score_edit(target, fake_seq_onehot)))

        best_score = None
        for i, start in enumerate(range(0, fake_seq_onehot.shape[2], datavar.seqs.shape[2])):
            stop = start + datavar.seqs.shape[2]
            fake_pred_slice = fake_pred[:, :, min_length * i // 4:min_length * (i + 1) // 4]
            assert fake_pred_slice.shape[2] in [min_length // 4, min_length // 4 + 1]
            score = fake_pred_slice.sigmoid().mean()
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i
                fake_seq_slice = fake_seq_onehot[:, :, start:stop]
                best_discriminator_score = float(score)
                best_blosum62_score = float(
                    pagnn.utils.score_blosum62(seqs[0, :, :].data, fake_seq_slice))
                best_edit_score = float(pagnn.utils.score_edit(seqs[0, :, :].data, fake_seq_slice))
                best_sequence = ''.join(pagnn.AMINO_ACIDS[int(i)]
                                        for i in np.argmax(pagnn.to_numpy(fake_seq_slice), 1)[0])
            start = stop

        assert i == (args.batch_size - 1)
        df_columns['best_discriminator_score'].append(best_discriminator_score)
        df_columns['best_blosum62_score'].append(best_blosum62_score)
        df_columns['best_edit_score'].append(best_edit_score)
        df_columns['sequence'].append(best_sequence)
        df_columns['sequence_type'].append('gen')
        df_columns['best_idx'].append(best_idx)

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
