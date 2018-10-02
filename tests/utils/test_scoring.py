import numpy as np
import pytest
import torch

from pagnn import settings
from pagnn.utils.array_ops import argmax_onehot
from pagnn.utils.scoring import score_blosum62, score_edit
from pagnn.utils.testing import set_device

# from pagnn.utils.dataset_ops import AMINO_ACIDS


@pytest.fixture(params=[False, True])
def use_cuda(request):
    return request.param


@pytest.fixture(params=[False, True])
def use_argmax(request):
    return request.param


@pytest.fixture(params=[64])
def batch_size(request):
    return request.param


@pytest.fixture(params=[256, 512, 1024, 2048])
def seq_length(request):
    return request.param


def test_score_blosum62_0(benchmark, use_cuda, use_argmax, batch_size, seq_length):
    target = torch.from_numpy(np.zeros((20, seq_length), dtype=np.float32)).to(settings.device)
    decoys = torch.from_numpy(np.zeros((batch_size, 20, seq_length), dtype=np.float32)).to(
        settings.device
    )
    if use_argmax:
        decoys = argmax_onehot(decoys)
    with set_device("cuda" if use_cuda else "cpu"):
        blosum62_score = benchmark(score_blosum62, target, decoys)
    assert blosum62_score == 0


def test_score_blosum62_1(benchmark, use_cuda, use_argmax, batch_size, seq_length):
    target = torch.from_numpy(np.zeros((20, seq_length), dtype=np.float32)).to(settings.device)
    target[1, 0] = 1
    decoys = torch.from_numpy(np.zeros((batch_size, 20, seq_length), dtype=np.float32)).to(
        settings.device
    )
    decoys[0, 3, 0] = 1  # V -> L : 1
    if use_argmax:
        decoys = argmax_onehot(decoys)
    with set_device("cuda" if use_cuda else "cpu"):
        blosum62_score = benchmark(score_blosum62, target, decoys)
    assert blosum62_score == 1 / seq_length


def test_score_blosum62_2(benchmark, use_cuda, use_argmax, batch_size, seq_length):
    target = torch.from_numpy(np.zeros((20, seq_length), dtype=np.float32)).to(settings.device)
    target[0, 0] = 1  # G
    target[0, 0] = 1  # L
    decoys = torch.from_numpy(np.zeros((batch_size, 20, seq_length), dtype=np.float32)).to(
        settings.device
    )
    decoys[0, 1, 0] = 1  # G -> V : -3
    decoys[0, 3, 1] = 1  # G -> L : -4
    decoys[1, 0, 0] = 1  # G -> G : 6
    decoys[1, 1, 1] = 1  # G -> A : 0
    if use_argmax:
        decoys = argmax_onehot(decoys)
    with set_device("cuda" if use_cuda else "cpu"):
        blosum62_score = benchmark(score_blosum62, target, decoys)
    assert blosum62_score == 6 / seq_length


def test_score_edit_0(benchmark, use_cuda, use_argmax, batch_size, seq_length):
    target = torch.from_numpy(np.zeros((20, seq_length), dtype=np.float32)).to(settings.device)
    decoys = torch.from_numpy(np.zeros((batch_size, 20, seq_length), dtype=np.float32)).to(
        settings.device
    )
    if use_argmax:
        decoys = argmax_onehot(decoys)
    with set_device("cuda" if use_cuda else "cpu"):
        blosum62_score = benchmark(score_edit, target, decoys)
    assert blosum62_score == 0


def test_score_edit_1(benchmark, use_cuda, use_argmax, batch_size, seq_length):
    target = torch.from_numpy(np.zeros((20, seq_length), dtype=np.float32)).to(settings.device)
    target[0, 0] = 1
    decoys = torch.from_numpy(np.zeros((batch_size, 20, seq_length), dtype=np.float32)).to(
        settings.device
    )
    decoys[0, 0, 0] = 1
    if use_argmax:
        decoys = argmax_onehot(decoys)
    with set_device("cuda" if use_cuda else "cpu"):
        edit_score = benchmark(score_edit, target, decoys)
    assert edit_score == 1 / seq_length
