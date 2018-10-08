import numpy as np
import pytest
from scipy import sparse

from pagnn import utils

EXPAND_ADJACENCY_TEST_DATA = [
    # (adj, expanded_adj)
    (np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int32),
     np.array(
         [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
          [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
         dtype=np.int32))
]


@pytest.mark.parametrize("adj, expanded_adj_", EXPAND_ADJACENCY_TEST_DATA)
def test_expand_adjacency(adj, expanded_adj_):
    np.array_equal(utils.expand_adjacency(sparse.coo_matrix(adj)).to_dense(), expanded_adj_)
