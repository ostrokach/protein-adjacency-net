import random
import unittest.mock

import numpy as np
import pytest

import pagnn


@pytest.mark.parametrize("original_seq, parmuted_seq",
                         [('ABC', 'BCA'),
                          ('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'MNOPQRSTUVWXYZABCDEFGHIJKL'),
                          (b'ABCDEFGHIJKLMNOPQRSTUVWXYZ', b'MNOPQRSTUVWXYZABCDEFGHIJKL')])
def test_permute_sequence(original_seq, parmuted_seq):
    rs = random.Random(42)
    with unittest.mock.patch('pagnn.dataset.random', rs):
        permuted_seq_ = pagnn.dataset.permute_sequence(original_seq)
    assert parmuted_seq == permuted_seq_


@pytest.mark.parametrize(
    "positive_seq, negative_seq, interpolate, interpolated_seqs, interpolated_targets",
    [(b'', b'', 0, [], []), (b'ABCDEF', b'HIJKLM', 1, [b'AIJKEF'], [0.5])])
def test_interpolate_sequences(positive_seq, negative_seq, interpolate, interpolated_seqs,
                               interpolated_targets):
    rs = np.random.RandomState(42)
    with unittest.mock.patch('pagnn.dataset.np.random', rs):
        interpolated_seqs_, interpolated_targets_ = pagnn.dataset.interpolate_sequences(
            positive_seq, negative_seq, interpolate)
    assert interpolated_seqs_ == interpolated_seqs
    assert interpolated_targets_ == interpolated_targets
