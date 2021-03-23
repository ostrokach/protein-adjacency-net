#!/bin/bash

set -ex


# Convert all Jupyter notebooks to RST skeletons and HTML files
for notebook in ../notebooks/*.ipynb ; do
  echo $notebook
  NOTEBOOK_NAME=$(basename ${notebook%.ipynb})
  jupyter nbconvert --to=html_ch --output=$NOTEBOOK_NAME.ipynb --output-dir=notebooks/ --template=output_toggle.tpl $notebook
  cat <<EOF > notebooks/$NOTEBOOK_NAME.ipynb.rst
$NOTEBOOK_NAME.ipynb
--------------------------------------------------------------------------------

.. raw:: html
   :file: $NOTEBOOK_NAME.ipynb.html
EOF
done


# Automatically generate 'notebooks.rst' listing all notebooks
cat <<EOF > notebooks.rst
Notebooks
=========

.. toctree::
    :maxdepth: 2
    :caption: Notebooks

EOF
for file in notebooks/*.rst ; do
    echo "    ${file%%.rst}" >> notebooks.rst
done

