import numpy as np
import pytest

from pagnn import faker

COUNT_MATCHES_TEST_DATA = [
    # (matrix, conv_filter, match_count)
    (
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
        1,
    ),
    (
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
        1,
    ),
    (
        np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
        0,
    ),
    (
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
        1,
    ),
]


@pytest.mark.parametrize("matrix, conv_filter, match_count_", COUNT_MATCHES_TEST_DATA)
def test_count_matches(matrix, conv_filter, match_count_):
    assert faker.count_matches(matrix, conv_filter) == match_count_
