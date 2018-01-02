#!/bin/bash
# The first argument should be the directory containing jupyter notebooks.
# The second argument should be the directory where to store produced html files.

set -ev

DIR="$(dirname "$(readlink -f "$0")")"

# Convert all Jupyter notebooks to RST skeletons and HTML files
for notebook in $1/*.ipynb ; do
  echo $notebook
  NOTEBOOK_NAME=$(basename ${notebook%.ipynb})
  jupyter nbconvert --to=html --output=${NOTEBOOK_NAME} --output-dir="$2/" --template="${DIR}/output_toggle.tpl" $notebook
done
