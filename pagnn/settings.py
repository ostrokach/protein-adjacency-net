"""Settings.

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

#:
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Number of amino acids to add between sequences when doing contract / expand operation.
GAP_LENGTH: int = 0

#: Skip over sequences that are less then ``MIN_SEQUENCE_LENGTH`` amino acids long.
MIN_SEQUENCE_LENGTH: int = 20

#: Profiler to use.
PROFILER = os.getenv("PROFILER")
assert PROFILER in [None, "cProfile", "line_profiler"]


def _is_array_job():
    array_job_id = (
        os.getenv("SGE_TASK_ID") or os.getenv("PBS_ARRAYID") or os.getenv("SLURM_ARRAY_TASK_ID")
    )
    return array_job_id is not None and int(array_job_id) > 1


#: ``True`` if we are running an array job and this is not the first job in the array.
ARRAY_JOB: bool = _is_array_job()


def _show_progressbar():
    return sys.stderr.isatty()


#: ``True`` if we should show a progressbar for training / validation.
SHOW_PROGRESSBAR: bool = _show_progressbar()
