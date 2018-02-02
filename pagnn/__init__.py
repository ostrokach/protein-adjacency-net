"""Top-level package for Protein Adjacency Graph Neural Network."""

__author__ = """Alexey Strokach"""
__email__ = 'alex.strokach@utoronto.ca'
__version__ = '0.1.3.dev0'
__all__ = ['scripts']

from . import *
from .config import *
from .utils import *
from .io import *
from .dataset import *
from .models import *
from .training import *
