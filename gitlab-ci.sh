#!/bin/bash

set -ex

if [[ \
      -z $1 ]] ; then
    echo "You must specify the test to run!"
    exit -1
fi

CI_PROJECT_NAME="$(basename $(pwd))"
CI_PROJECT_NAMESPACE="$(basename $(dirname $(pwd)))"

echo "${CI_PROJECT_NAME}"

gitlab-runner exec docker \
    --env CI_PROJECT_NAME="${CI_PROJECT_NAME}" \
    --env DATABASE_DATA_DIR="${DATABASE_DATA_DIR}" \
    --env DATABIN_DIR="${DATABIN_DIR}" \
    --docker-volumes "$HOME/.ssh/gitlab_ci_rsa:/root/.ssh/id_rsa:ro" \
    --docker-volumes "$(pwd):/home/${CI_PROJECT_NAME}:rw" \
    --docker-volumes "${DATABASE_DATA_DIR}:${DATABASE_DATA_DIR}:rw" \
    --docker-volumes "${DATABIN_DIR}:${DATABIN_DIR}:rw" \
    $1
