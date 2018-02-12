#!/bin/bash

set -ev

cd "${RECIPE_DIR}/.."
cat setup.cfg
python -m isort -c || true
python -m flake8 || true
python -m mypy -p ${PKG_NAME} || true
python -m pytest --cov="${SP_DIR}/${PKG_NAME}"
