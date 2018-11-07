#!/bin/bash

set -ev

REQUIRED_VARS=(
  OUTPUT_DIR
  DATAPKG_OUTPUT_DIR
  NETWORK_NAME
  SLURM_ARRAY_TASK_ID
  SBATCH_TIMELIMIT
)

for var in ${REQUIRED_VARS[*]} ; do
  if [[ -z "${!var}" ]] ; then
    echo "Environment variable '${var}' has not been set!"
    exit -1
  fi
done


sed "s|class Custom(nn.Module):|class DCN_${NETWORK_NAME}(nn.Module):|" ./src/model.py | \
  sed "s|pagnn.models.dcn.Custom = Custom|pagnn.models.dcn.DCN_${NETWORK_NAME} = DCN_${NETWORK_NAME}|" > \
  "${OUTPUT_DIR}/model.py"


python -m pagnn.training.dcn \
  --root-path "${OUTPUT_DIR}" \
  --training-data-path "${DATAPKG_OUTPUT_DIR}/adjacency-net-v2/master/training_dataset/adjacency_matrix.parquet" \
  --gpu -1 \
  --verbosity 1 \
  --network-name "DCN_${NETWORK_NAME}" \
  --custom-module "${OUTPUT_DIR}/model.py" \
  --num-negative-examples 63 \

# --training-data-cache "${DATAPKG_OUTPUT_DIR}/adjacency-net-v2/master/training_dataset/array_id_${SLURM_ARRAY_TASK_ID}"
