import numpy as np
import pytest

from pagnn import utils


@pytest.mark.parametrize(
    "positive_seq, negative_seq, interpolate, interpolated_seqs, interpolated_targets",
    [(b"", b"", 0, [], []), (b"ABCDEF", b"HIJKLM", 1, [b"ABJKLF"], [0.5])],
)
def test_interpolate_sequences(
    positive_seq, negative_seq, interpolate, interpolated_seqs, interpolated_targets
):
    random_state = np.random.RandomState(42)
    interpolated_seqs_, interpolated_targets_ = utils.interpolate_sequences(
        positive_seq, negative_seq, interpolate, random_state
    )
    assert interpolated_seqs_ == interpolated_seqs
    assert interpolated_targets_ == interpolated_targets
