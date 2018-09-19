#!/usr/bin/env bash

set -e

conda env update -n ${CONDA_ENV_NAME} -f environment.yaml

source activate ${CONDA_ENV_NAME}

jupyter nbextension enable collapsible_headings/main
jupyter nbextension enable runtools/main
jupyter nbextension enable codefolding/main
jupyter nbextension enable code_prettify/isort
jupyter nbextension enable toc2/main
