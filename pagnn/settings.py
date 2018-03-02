"""Module settings.

.. rubric:: Constants

.. autosummary::

   CUDA
   GAP_LENGTH
   MIN_SEQUENCE_LENGTH
   ARRAY_JOB
   SHOW_PROGRESSBAR
"""
import os
import sys

import torch

CUDA = torch.cuda.is_available()
"""We are using a CUDA GPU."""

GAP_LENGTH = 0
"""."""

MIN_SEQUENCE_LENGTH = 20
"""."""


def _is_array_job():
    array_job_id = (os.getenv('SGE_TASK_ID') or os.getenv('PBS_ARRAYID') or
                    os.getenv('SLURM_ARRAY_TASK_ID'))
    return array_job_id is not None and int(array_job_id) > 1


ARRAY_JOB = _is_array_job()
"""We are running an array job and it is not the first array job."""


def _show_progressbar():
    return sys.stderr.isatty()


SHOW_PROGRESSBAR = _show_progressbar()
"""Show progress bar for training / validation."""
