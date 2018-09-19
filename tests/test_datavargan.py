import random
from contextlib import contextmanager

import numpy as np
import pytest
from scipy import sparse
from torch.autograd import Variable

from pagnn import settings
from pagnn.datavargan import (
    conv2d_shape,
    dataset_to_datavar,
    pool_adjacency_mat,
    pool_adjacency_mat_reference,
    push_seqs,
)
from pagnn.types import DataSetGAN
from pagnn.utils import AMINO_ACIDS, to_numpy, to_sparse_tensor


@contextmanager
def no_cuda():
    cuda = settings.CUDA
    try:
        settings.CUDA = False
        yield
    finally:
        settings.CUDA = cuda


# === Fixtures ===


@pytest.fixture(
    params=random.Random(42).choices(
        [
            (dims, density)
            for dims in [10, 100, 200, 400, 800, 1600]
            for density in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        ],
        k=20,
    ),
    ids=lambda p: f"adj_{p[0]}_{p[1]}",
)
def adj(request):
    dims, density = request.param
    adj = (sparse.rand(dims, dims, density) + sparse.eye(dims)).tocoo()
    adj.data = np.ones(adj.nnz, dtype=np.int16)
    return adj


@pytest.fixture(
    params=[
        (num_seqs, seq_length) for num_seqs in [1, 64, 256, 1024] for seq_length in [50, 400, 800]
    ],
    ids=lambda p: f"ds_{p[0]}_{p[1]}",
)
def ds(request):
    num_seqs, seq_length = request.param
    random_state = np.random.RandomState(42)
    seqs = [
        "".join(AMINO_ACIDS[random_state.randint(0, 20)] for _ in range(seq_length)).encode("ascii")
        for _ in range(num_seqs)
    ]
    adj = (sparse.rand(seq_length, seq_length, 0.2) + sparse.eye(seq_length)).tocoo()
    adj.data = np.ones(adj.nnz, dtype=np.int16)
    return DataSetGAN(seqs=seqs, adjs=[adj], targets=[1], meta={})


# === Tests ===


def test_dataset_to_datavar_perf(benchmark, ds):
    with no_cuda():
        benchmark(dataset_to_datavar, ds)


def test_push_seq(benchmark):
    batch_size = 64
    seq_len = 600
    random_state = np.random.RandomState(0)
    indexes = random_state.randint(0, 20, batch_size)
    seqs = [AMINO_ACIDS[idx].encode("ascii") * seq_len for idx in indexes]
    # Actual
    with no_cuda():
        seqs_var = benchmark(push_seqs, seqs)
    # Referebce
    seqs_var_ = np.zeros((batch_size, 20, seq_len), dtype=float)
    for i, idx in enumerate(indexes):
        seqs_var_[i, idx, :] = 1
    # Compare
    assert np.allclose(seqs_var.data.numpy(), seqs_var_)


@pytest.mark.parametrize(
    "dims, density, kernel_size, stride, padding",
    random.Random(42).choices(
        [
            (dims, density, kernel_size, stride, padding)
            for dims in [10, 100, 200, 400, 800, 1600]
            for density in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
            for kernel_size in [3, 4, 5, 6, 7]
            for stride in [2, 3, 4]
            for padding in [0, 1, 2, 3]
        ],
        k=20,
    ),
)
def test_pool_adjacency_mat(benchmark, dims, density, kernel_size, stride, padding):
    adj = (sparse.rand(dims, dims, density) + sparse.eye(dims)).tocoo()
    adj.data = np.ones(adj.nnz, dtype=np.int16)

    shape = conv2d_shape(adj.shape, kernel_size, stride, padding)

    adj_pooled = pool_adjacency_mat_reference(
        Variable(to_sparse_tensor(adj).to_dense()), kernel_size, stride, padding
    )
    assert adj_pooled.shape[0] == shape[0]

    with no_cuda():
        adj_pooled_ = benchmark.pedantic(
            pool_adjacency_mat,
            args=(adj, kernel_size, stride, padding),
            rounds=1,
            iterations=1,
            warmup_rounds=1,
        )
    assert adj_pooled_.shape == shape

    assert np.allclose(to_numpy(adj_pooled), adj_pooled_.todense())


@pytest.mark.parametrize(
    "dims, density, kernel_size, stride, padding",
    [
        (dims, density, kernel_size, stride, padding)
        for dims in [10, 50, 100, 200, 400, 800, 1200]
        for density in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        for kernel_size in [4]
        for stride in [2]
        for padding in [1]
    ],
)
def test_pool_adjacency_mat_forplot(benchmark, dims, density, kernel_size, stride, padding):
    adj = (sparse.rand(dims, dims, density) + sparse.eye(dims)).tocoo()
    adj.data = np.ones(adj.nnz, dtype=np.int16)

    shape = conv2d_shape(adj.shape, kernel_size, stride, padding)

    adj_pooled = pool_adjacency_mat_reference(
        Variable(to_sparse_tensor(adj).to_dense()), kernel_size, stride, padding
    )
    assert adj_pooled.shape[0] == shape[0]

    with no_cuda():
        adj_pooled_ = benchmark.pedantic(
            pool_adjacency_mat,
            args=(adj, kernel_size, stride, padding),
            rounds=1,
            iterations=1,
            warmup_rounds=1,
        )
    assert adj_pooled.shape[0] == shape[0]

    assert np.allclose(to_numpy(adj_pooled), adj_pooled_.todense())
