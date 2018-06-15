#!/bin/bash

set -euv

unset XDG_RUNTIME_DIR

EXTENSION=""
if [[ -n ${SLURM_ARRAY_TASK_ID} ]] ; then
    EXTENSION="_${SLURM_ARRAY_TASK_ID}"
fi

# NB: Notebooks can execute for a week if neccessary
jupyter nbconvert ./notebooks/${NOTEBOOK_NAME}.ipynb \
    --to=html_ch \
    --execute \
    --output="${NOTEBOOK_NAME}${EXTENSION}.html" \
    --output-dir="${OUTPUT_DIR}" \
    --ExecutePreprocessor.timeout=$((60 * 60 * 24 * 7))
    # --template=docs/output_toggle.tpl \
