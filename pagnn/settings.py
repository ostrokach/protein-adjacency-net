import os

import torch

CUDA = torch.cuda.is_available()
GAP_LENGTH = 0
MIN_SEQUENCE_LENGTH = 20


def _is_array_job():
    array_job_id = (os.getenv('SGE_TASK_ID') or os.getenv('PBS_ARRAYID') or
                    os.getenv('SLURM_ARRAY_TASK_ID'))
    return array_job_id is not None and int(array_job_id) > 1


ARRAY_JOB: bool = _is_array_job()
