#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=62G
#SBATCH --account=def-pmkim
#SBATCH --job-name=run-notebook-cpu
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=alexey.strokach@kimlab.org
#SBATCH --output=/scratch/p/pmkim/strokach/tmp/log/run-notebook-cpu-%N-%j.log

unset XDG_RUNTIME_DIR

mkdir ${SLURM_TMPDIR}/env
tar -xzf ~/datapkg_data_dir/conda-envs/defaults/defaults-v23.tar.gz -C ${SLURM_TMPDIR}/env

pushd /dev/shm
ln -s ${SLURM_TMPDIR}/env
popd

source /dev/shm/env/bin/activate
conda-unpack
# source /dev/shm/env/bin/deactivate

# pushd ~/workspace/proteinsolver
# python -m pip install -e . --no-deps --no-index --no-cache-dir --disable-pip-version-check --no-use-pep517
# popd

# conda activate base
# jupyter lab --ip 0.0.0.0 --no-browser

NOTEBOOK_STEM=$(basename ${NOTEBOOK_PATH%%.ipynb})
NOTEBOOK_DIR=$(dirname ${NOTEBOOK_PATH})
OUTPUT_TAG="${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_JOB_ID}-${SLURM_JOB_ID}-${SLURM_JOB_NODELIST}"

mkdir -p "${NOTEBOOK_DIR}/${NOTEBOOK_STEM}/logs"
papermill --no-progress-bar --log-output --kernel python3 "${NOTEBOOK_PATH}" "${NOTEBOOK_DIR}/${NOTEBOOK_STEM}/logs/${NOTEBOOK_STEM}-${OUTPUT_TAG}.ipynb"

# sleep 72h
