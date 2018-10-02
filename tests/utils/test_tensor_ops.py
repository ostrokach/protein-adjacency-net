from pagnn.utils import set_device
from pagnn.utils.tensor_ops import argmax_onehot

from .test_array_ops import seq  # noqa


def test_argmax_onehot(benchmark, seq):  # noqa
    with set_device("cuda" if seq.is_cuda else "cpu"):
        seq_onehot = benchmark(argmax_onehot, seq)
    assert (seq == seq_onehot).all()
