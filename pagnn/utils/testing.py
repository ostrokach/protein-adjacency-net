from contextlib import contextmanager

from pagnn import settings


@contextmanager
def set_cuda(use_cuda):
    _cuda = settings.CUDA
    try:
        settings.CUDA = use_cuda
        yield
    finally:
        settings.CUDA = _cuda
