#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=rrg-pmkim
#SBATCH --job-name=pagnn-validation
#SBATCH --export=ALL

set -ev

unset XDG_RUNTIME_DIR

if [[ -z ${MY_JOB_IDX} ]] ; then
    echo "MY_JOB_IDX environment variable has to be set!"
    exit 1
fi

python ~/working/pagnn/pagnn/prediction/gan/validation.py \
    --input-file ./1vkf_adjacencies.parquet \
    --output-file ./validation${MY_JOB_IDX}.parquet \
    --work-path ~/datapkg/adjacency-net/notebooks/train_neural_network/gan-permute-seq-0-test_x12-0.1.9.dev-4a07eef/ \
    --step 5900 \
    --with-ss \
    --nseqs 1000