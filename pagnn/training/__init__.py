"""Scripts for training models.

.. autosummary::
   :toctree: _modules

   common
   dcn_old
   dcn
   gan
"""
__all__ = ["dcn_old", "dcn", "gan"]
from .common import *
from . import *
