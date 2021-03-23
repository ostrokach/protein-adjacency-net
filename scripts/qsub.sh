#!/bin/bash

set -ev

unset XDG_RUNTIME_DIR

# === Spark jobs ===
if [[ -n ${USE_SPARK} ]] ; then
    export SPARK_MASTER_HOST=$(hostname -f)
    export SPARK_IDENT_STRING=${SLURM_JOBID}

    #: Location where spark workers store log files, etc.
    export SPARK_WORKER_DIR=${SLURM_TMPDIR}/spark-work
    #: Location that Spark uses for overflowing RDDs.
    export SPARK_LOCAL_DIRS=${SLURM_TMPDIR}/spark-tmp
    #: Location for storing executor log files.
    export SPARK_LOG_DIR=${OUTPUT_DIR}/spark-logs
    mkdir -p ${SPARK_WORKER_DIR} ${SPARK_LOCAL_DIRS} ${SPARK_LOG_DIR}

    export SPARK_DAEMON_MEMORY=1g
    # export SPARK_DAEMON_JAVA_OPTS="-DXms=1g -DXmx=120g"

    # --- Start Spark master ---
    ${SPARK_HOME}/sbin/start-master.sh
    sleep 20

    # --- Start Spark workers ---
    export SPARK_NO_DAEMONIZE=1

    # For some reason, SLURM_MEM_PER_NODE does not respect the total amount of memory
    export MEM_ON_NODE=$(awk '/MemTotal/ { printf "%.0f \n", $2/1024 }' /proc/meminfo)

    srun -w $(hostname -s) \
        -N 1 -n 1 -c ${SLURM_CPUS_ON_NODE} --mem=0 \
        --output=${SPARK_LOG_DIR}/spark-worker-${CI_JOB_ID}-%N-%j.out \
        ${SPARK_HOME}/sbin/start-slave.sh spark://${SPARK_MASTER_HOST}:7077 \
        --cores $((SLURM_CPUS_ON_NODE - 1)) \
        --memory $((SLURM_MEM_PER_NODE - 65000))M &

    if [[ "${SLURM_NTASKS}" -gt "1" ]] ; then
        srun -x $(hostname -s) \
            -N $((SLURM_NTASKS - 1)) -n $((SLURM_NTASKS - 1)) -c ${SLURM_CPUS_ON_NODE} --mem=0 \
            --output=${SPARK_LOG_DIR}/spark-worker-${CI_JOB_ID}-%N-%j.out \
            ${SPARK_HOME}/sbin/start-slave.sh spark://${SPARK_MASTER_HOST}:7077 \
            --cores ${SLURM_CPUS_ON_NODE} \
            --memory ${SLURM_MEM_PER_NODE}M &
    fi

    sleep 40
fi

# === Array jobs ===
EXTENSION=""
if [[ -n ${SLURM_ARRAY_TASK_ID} ]] ; then
    EXTENSION="_${SLURM_ARRAY_TASK_ID}"
fi

# === Run notebook ===

# TODO: Move this to a standalone package
if [[ -n ${INTERACTIVE} ]] ; then
    jupyter notebook
else
    python ${SLURM_SUBMIT_DIR}/scripts/execute_notebook.py \
        -i ./notebooks/${NOTEBOOK_NAME}.ipynb \
        -o "${OUTPUT_DIR}/notebooks/${NOTEBOOK_NAME}${EXTENSION}.html"
fi

# Alternatively, we can use the CLI
# jupyter nbconvert ./notebooks/${NOTEBOOK_NAME}.ipynb \
#     --to=html_ch \
#     --execute \
#     --allow-errors \
#     --output="${NOTEBOOK_NAME}${EXTENSION}.html" \
#     --output-dir="${OUTPUT_DIR}" \
#     --ExecutePreprocessor.timeout=$((60 * 60 * 24 * 7))
#     # --template=docs/output_toggle.tpl \

# === Cleanup ===
if [[ -n ${USE_SPARK} ]] ; then
    ${SPARK_HOME}/sbin/stop-master.sh
fi
