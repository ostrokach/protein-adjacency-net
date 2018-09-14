#!/bin/bash

set -ev

python -m pytest \
    -c setup.cfg \
    --cov="${SP_DIR}/pagnn" \
    --benchmark-disable \
    --color=yes \
    tests/

sed -i "s|${SP_DIR}||g" .coverage
mv .coverage "${PACKAGE_ROOT_DIR}/.coverage"
