import argparse

from pagnn import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        '--rootdir',
        type=str,
        default='.',
        help="Location where to create subfolders for storing network data and cache files.",
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='.',
        help="Location of the `adjacency-net` databin folder.",
    )
    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help="Number of negative sequences per batch.",
    )
    parser.add_argument(
        '--steps-between-validation',
        type=int,
        default=1000,
        help="Number of negative sequences per batch.",
    )
    parser.add_argument(
        '--d-iters',
        type=int,
        default=64,
        help="Number of D iters per each G iter.",
    )
    # ...
    parser.add_argument(
        '--adam',
        action='store_true',
        help='Whether to use adam (default is RMSprop)',
    )
    parser.add_argument(
        '--learning_rate_d',
        type=float,
        default=0.00005,
        help="Learning rate for Discriminator (Critic).",
    )
    parser.add_argument(
        '--learning_rate_g',
        type=float,
        default=0.00005,
        help="Learning rate for Generator.",
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help="beta1 for Adam",
    )
    parser.add_argument(
        '--clamp_lower',
        type=float,
        default=-0.01,
        help="WGAN requires that you clamp the weights.",
    )
    parser.add_argument(
        '--clamp_upper',
        type=float,
        default=0.01,
        help="WGAN requires that you clamp the weights.",
    )
    # Network parameters
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
    parser.add_argument('--resume', action='store_true', default=settings.ARRAY_JOB)
    parser.add_argument('--num-aa-to-process', type=int, default=None)
    # Visuals
    parser.add_argument('--progressbar', action='store_true', default=settings.SHOW_PROGRESSBAR)
    # TODO(AS): Run concurrent jobs in the computer has multiple GPUs
    parser.add_argument('-n', '--num-concurrent-jobs', type=int, default=1)
    args = parser.parse_args()
    return args
