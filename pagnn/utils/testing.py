from contextlib import contextmanager

import numpy as np
import torch

from pagnn import settings


@contextmanager
def set_cuda(use_cuda):
    _cuda = settings.CUDA
    try:
        settings.CUDA = use_cuda
        yield
    finally:
        settings.CUDA = _cuda


def random_sequence(batch_size, seq_size, seq_length, random_state=None):
    """

    Args:
        batch_size: Number of different sequences of the same size.
        seq_size: Number of amino acids that are posible at each residue.
        seq_length: Length of each sequence.

    Returns:
        Generated PyTorch sequence.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    seq_idxs = random_state.randint(0, seq_size, size=(batch_size, 1, seq_length))
    seq = torch \
        .zeros(batch_size, seq_size, seq_length) \
        .scatter_(1, torch.from_numpy(seq_idxs), 1)

    assert seq.shape == (batch_size, seq_size, seq_length)
    assert (seq.sum(1) == 1).all()

    return seq
