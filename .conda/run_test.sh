#!/bin/bash

set -ev

cd "${RECIPE_DIR}/.."
python -m isort -c
python -m flake8
python -m mypy -p ${PKG_NAME}
python -m pytest --cov="${SP_DIR}/${PKG_NAME}"
