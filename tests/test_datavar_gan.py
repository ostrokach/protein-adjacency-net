import numpy as np
from contextlib import contextmanager
from pagnn import settings
from pagnn.datavar_gan import push_seqs


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
        seqs_var = push_seqs(seqs).numpy()
    # Ref
    seqs_var_ = np.zeros((3, 20, 5), dtype=float)
    seqs_var_[0, 0, :] = 1
    seqs_var_[1, 1, :] = 1
    seqs_var_[2, 2, :] = 1
    #
    assert np.allclose(seqs_var, seqs_var_)
