import numpy as np
from numba import float32, int32, int64, jit
from scipy import stats


@jit(float32[:, :](int64, int64), nopython=True)
def generate_data(num_points: int, num_features: int) -> np.array:
    """Generate example data for testing adjacency convolutions.

    Args:
        num_points: Number of points (x-axis).
        num_features: Number of features (y-axis).

    Returns:
        NumPy array of randomly-generated data, with one ``1`` per row.
    """
    data = np.zeros((num_points, num_features), dtype=np.float32)
    feature_idxs = np.arange(num_features)
    for i in np.arange(num_points):
        data[i, np.random.choice(feature_idxs)] = 1.0
    return data


@jit(int32(float32[:, :], float32[:, :]), nopython=True)
def count_matches(data: np.array, conv_filter: np.array) -> int:
    """Count the number of times `conv_filter` is inside `data`.

    Args:
        data: Strided adjacency matrix (*two-rows-per-interaction*).
        conv_filter: Convolution to apply on the adjacency matrix.

    Returns:
        Number of times the convolution filter ``conv_filter`` appears in input data.
    """
    convs = []
    for i in np.arange(0, data.shape[0], 2):
        conv = (data[i:i + 2, :] * conv_filter).sum()
        convs.append(conv)
    return convs.count(2)


def label_data(data: np.array, conv_filter: np.array, cutoff_proba: float) -> bool:
    """Classify data based on the number of occurences of `conv_filter`.

    Note:
        Effective `cutoff_proba` can be up to 5% less strict because of ``>=``.

    Args:
        data: Strided adjacency matrix (*two-rows-per-interaction*).
        conv_filter: Convolution to apply on the adjacency matrix.
        cutoff_proba: Fraction of randomly-generated data that should be labelled as real.

    Returns:
        ``True`` if the protein is predicted to be real. ``False`` otherwise.
    """
    num_matches = count_matches(data, conv_filter)
    cutoff = stats.binom(data.shape[0] / 2, 1 / 25).ppf(cutoff_proba)
    return num_matches >= cutoff


def expand_adjacency(adj: np.array) -> np.array:
    """Convert adjacency matrix into a strided mask.

    Args:
        adj: Adjacency matrix.

    Returns:
        Adjacency matrix converted to *two-rows-per-interaction* format.
    """
    new_adj = np.zeros((int(adj.sum() * 2), adj.shape[1]), dtype=adj.dtype)
    idx = 0
    for x, y in zip(*adj.nonzero()):
        new_adj[idx, x] = 1
        new_adj[idx + 1, y] = 1
        idx += 2
    return new_adj
