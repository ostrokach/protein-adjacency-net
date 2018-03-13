import numpy as np
import pytest
import torch

from pagnn.utils.array_ops import argmax_onehot
from pagnn.utils.testing import set_cuda


@pytest.fixture(scope='module', params=[False, True], ids=lambda r: f'seq_cuda{int(r.cuda)}')
def seq(request):
    random_state = np.random.RandomState(42)
    use_cuda = request.param
    seq = torch.zeros(
        64, 20, 512, out=torch.cuda.FloatTensor() if use_cuda else torch.FloatTensor())
    for i in range(seq.shape[0]):
        for j in range(seq.shape[2]):
            idx = random_state.randint(0, 20)
            seq[i, idx, j] = 1
    assert seq.is_cuda == use_cuda
    return seq


def test_argmax_onehot(benchmark, seq):
    with set_cuda(seq.is_cuda):
        seq_onehot = benchmark(argmax_onehot, seq)
    assert (seq == seq_onehot).all()
