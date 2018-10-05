import numpy as np
import pytest
import torch
from scipy import sparse

from pagnn.types import DataSetGAN
from pagnn.utils import seq_to_array


@pytest.mark.parametrize(
    "seq_bytes_list, meta",
    [
        #
        (seq, None)
        for seq in [[b"A" * 12_000], [b""], [b"AAA", b"CCC", b"EEE"]]
    ]
    + [
        #
        ([b"ACDDFFA"], meta)
        for meta in [{}, {"exp": [1, 2, 3], "act": None, "val": 1.4}, None]
    ],
)
def test_datasetgan_serde(seq_bytes_list, meta, benchmark):
    random_state = np.random.RandomState(42)
    seqs = [seq_to_array(seq_bytes) for seq_bytes in seq_bytes_list]
    adjs = [
        sparse.coo_matrix(
            (random_state.rand(seq.shape[1], seq.shape[1]) + np.eye(seq.shape[1])) > 0.8,
            dtype=np.float,
        )
        for seq in seqs
    ]
    targets = torch.ones(len(seqs), dtype=torch.float)

    @benchmark
    def roundtrip():
        datasetgan = DataSetGAN(seqs, adjs, targets, meta)
        buf = datasetgan.to_buffer()
        datasetgan_ = DataSetGAN.from_buffer(buf)
        assert datasetgan == datasetgan_
