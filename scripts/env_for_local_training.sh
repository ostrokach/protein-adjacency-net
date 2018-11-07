export SLURM_ARRAY_TASK_ID="0"
export SBATCH_TIMELIMIT="1:00:00"
export NETWORK_NAME=$(git rev-parse --abbrev-ref HEAD)
export OUTPUT_DIR="/tmp/strokach/${NETWORK_NAME}"
mkdir -p ${OUTPUT_DIR}

