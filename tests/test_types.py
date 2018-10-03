import numpy as np
import pytest
import torch
from scipy import sparse

from pagnn.types import DataSetGAN
from pagnn.utils import seq_to_array


@pytest.mark.parametrize(
    "seq_bytes_list, meta",
    [
        (seq, meta)
        for seq in [[b"AAAAA"], [b"ACACAPPPCGGD"], [b""], [b"AAA", b"CCC", b"EEE"]]
        for meta in [{}, {"exp": [1, 2, 3], "act": None, "val": 1.4}, None]
    ],
)
def test_datasetgan_serde(seq_bytes_list, meta):
    seqs = [seq_to_array(seq_bytes) for seq_bytes in seq_bytes_list]
    adjs = [sparse.eye(seq.shape[1], dtype=np.float).tocoo() for seq in seqs]
    targets = torch.ones(len(seqs))

    datasetgan = DataSetGAN(seqs, adjs, targets, meta)
    buf = datasetgan.to_buffer()
    datasetgan_ = DataSetGAN.from_buffer(buf)

    assert datasetgan == datasetgan_
