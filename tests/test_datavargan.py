from contextlib import contextmanager

import numpy as np

from pagnn import datavargan, settings


@contextmanager
def no_cuda():
    cuda = settings.CUDA
    try:
        settings.CUDA = False
        yield
    finally:
        settings.CUDA = cuda


def test_push_seq():
    seqs = [b'GGGGG', b'VVVVV', b'AAAAA']
    with no_cuda():
        seqs_var = datavargan._push_seqs(seqs).data.numpy()
    # Ref
    seqs_var_ = np.zeros((3, 20, 5), dtype=float)
    seqs_var_[0, 0, :] = 1
    seqs_var_[1, 1, :] = 1
    seqs_var_[2, 2, :] = 1
    #
    assert np.allclose(seqs_var, seqs_var_)
