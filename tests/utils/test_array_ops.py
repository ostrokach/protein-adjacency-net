import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy import sparse

from pagnn.utils.array_ops import (add_eye_sparse, argmax_onehot, remove_eye, remove_eye_sparse,
                                   unfold_from, unfold_to)
from pagnn.utils.testing import random_sequence, set_cuda


@pytest.fixture(
    scope='module',
    params=[False] + ([True] if torch.cuda.is_available() else []),
    ids=lambda r: f'seq_cuda{int(r.cuda)}')
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


def test_unfold():
    in_features = 20
    out_features = 20
    ln = nn.Linear(in_features, out_features, bias=False)
    ln.weight.data.set_(torch.eye(in_features, out_features))
    seq = random_sequence(1, 5, 4)
    seq2 = unfold_to(seq, 20)
    seq3 = ln(seq2)
    seq4 = unfold_from(seq3, 5)
    assert (seq == seq4).all()


@pytest.mark.parametrize(
    "size, bandwidth",
    [(size, bandwidth) for size in range(3, 20) for bandwidth in range(0, 4) if bandwidth <= size])
def test_remove_eye(size, bandwidth):
    ar = np.ones((size, size))
    ar_noeye = remove_eye(ar, bandwidth)

    for i in range(bandwidth):
        assert ar[0, i] == 1
        assert ar_noeye[0, i] == 0

    spar = sparse.coo_matrix(ar)
    spar_noeye = remove_eye_sparse(spar, bandwidth)

    assert np.allclose(ar_noeye, spar_noeye.todense())


@pytest.mark.parametrize(
    "size, bandwidth",
    [(size, bandwidth) for size in range(3, 20) for bandwidth in range(0, 4) if bandwidth <= size])
def test_add_eye(size, bandwidth):
    ar = np.ones((size, size))
    spar = sparse.coo_matrix(ar)
    spar_noeye = remove_eye_sparse(spar, bandwidth)
    spar_eye = add_eye_sparse(spar_noeye, bandwidth)
    assert np.allclose(ar, spar_eye.todense())


@pytest.mark.parametrize(
    "size, bandwidth",
    [(size, bandwidth) for size in range(3, 20) for bandwidth in range(1, 4) if bandwidth <= size])
def test_add_eye_inplace(size, bandwidth):
    ar = np.ones((size, size))
    spar = sparse.coo_matrix(ar)
    remove_eye_sparse(spar, bandwidth, copy=False)
    assert not np.allclose(ar, spar.todense())
    add_eye_sparse(spar, bandwidth, copy=False)
    assert np.allclose(ar, spar.todense())
