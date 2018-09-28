#!/bin/bash

set -ev

python -m pytest \
    -c setup.cfg \
    --cov="${SP_DIR}/pagnn" \
    --benchmark-disable \
    --color=yes \
    tests/

sed -i "s|${SP_DIR}||g" .coverage

pwd
mv .coverage "${RECIPE_DIR}/.coverage"
ls "${RECIPE_DIR}/.coverage"
