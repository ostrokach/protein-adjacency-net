import numpy as np
import pytest
from scipy import sparse

from pagnn import utils

COUNT_MATCHES_TEST_DATA = [
    # (matrix, conv_filter, match_count)
    (np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ], dtype=np.float32), np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32), 1),
    (np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.float32), np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32), 1),
    (np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ], dtype=np.float32), np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32), 0),
    (np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ], dtype=np.float32), np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32), 1),
]


@pytest.mark.parametrize("matrix, conv_filter, match_count_", COUNT_MATCHES_TEST_DATA)
def test_count_matches(matrix, conv_filter, match_count_):
    assert utils.count_matches(matrix, conv_filter) == match_count_


EXPAND_ADJACENCY_TEST_DATA = [
    # (adj, expanded_adj)
    (np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int32), np.array(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
         [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        dtype=np.int32))
]


@pytest.mark.parametrize("adj, expanded_adj_", EXPAND_ADJACENCY_TEST_DATA)
def test_expand_adjacency(adj, expanded_adj_):
    np.array_equal(utils.expand_adjacency(sparse.coo_matrix(adj)).todense(), expanded_adj_)
