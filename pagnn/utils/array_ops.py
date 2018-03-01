import numpy as np
import torch
from scipy import sparse
from torch.autograd import Variable

from pagnn import settings


def to_numpy(array: Variable) -> np.ndarray:
    """Convert torch `Variable` into a numpy `ndarray`."""
    if settings.CUDA:
        return array.data.cpu().numpy()
    else:
        return array.data.numpy()


def to_tensor(array: np.ndarray) -> torch.FloatTensor:
    """Convert a numpy `ndarray` into a torch tensor (possibly on CUDA)."""
    tensor = torch.FloatTensor(array)
    if settings.CUDA:
        tensor = tensor.cuda()
    return tensor


def to_sparse_tensor(sparray: sparse.spmatrix) -> torch.sparse.FloatTensor:
    """Convert a scipy `spmatrix` into a torch sparse tensor (possibly on CUDA)."""
    i = torch.LongTensor(np.vstack([sparray.row, sparray.col]))
    v = torch.FloatTensor(sparray.data)
    s = torch.Size(sparray.shape)
    tensor = torch.sparse.FloatTensor(i, v, s)
    if settings.CUDA:
        tensor = tensor.cuda()
    return tensor
