#!/bin/bash

set -ex

if [[ \
      -z $1 ]] ; then
    echo "You must specify the test to run!"
    exit -1
fi

rm -rf public
mkdir public
chmod ugo+rwX public

CI_PROJECT_NAME="$(basename $(pwd))"
CI_PROJECT_NAMESPACE="$(basename $(dirname $(pwd)))"

echo "${CI_PROJECT_NAME}"

gitlab-runner exec docker \
    --env CI_PROJECT_NAME="${CI_PROJECT_NAME}" \
    --docker-volumes "$HOME/.ssh/gitlab_ci_rsa:/root/.ssh/id_rsa:ro" \
    --docker-volumes "$(pwd):/home/${CI_PROJECT_NAME}:rw" \
    --docker-volumes "/kimlab2/database_data:/kimlab2/database_data:rw" \
    $1

