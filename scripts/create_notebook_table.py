#!/usr/bin/env python3
import csv
import json
import os
import os.path as op

import pypandoc


def parse_notebooks(notebook_dir):
    """Get notebook name and description for every notebook in `notebook_dir`."""
    notebooks = []
    for notebook in os.listdir(notebook_dir):
        if not notebook.endswith('.ipynb'):
            continue
        notebook_file = op.join(notebook_dir, notebook)
        with open(notebook_file) as fin:
            notebook_data = json.load(fin)
        first_cell_md = extract_first_cell_text(notebook_data)
        first_cell_rst = md_to_rst(first_cell_md) if first_cell_md else ""
        notebooks.append((notebook, first_cell_rst))
    return notebooks


def extract_first_cell_text(notebook_data):
    """Extract MarkDown data from first cell."""
    try:
        first_cell = notebook_data['cells'][0]
    except (AttributeError, IndexError):
        return ""

    if first_cell['cell_type'] != 'markdown':
        return ""

    return first_cell['source']


def md_to_rst(md_text):
    """Clean up MarkDown and convert to RestructuredText."""
    md_text = ' '.join(l.strip()
                       for l in md_text
                       if not l.startswith('#') and not l.startswith('---') if l.strip())
    return pypandoc.convert_text(md_text, 'rst', format='md')


def main(notebook_dir, output_file):
    notebooks = parse_notebooks(notebook_dir)
    notebooks.sort(key=lambda x: x[0])
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Name', 'Description'])
        for notebook_name, notebook_description in notebooks:
            notebook_url = '`{} <notebooks/{}.html>`_'.format(notebook_name,
                                                              op.splitext(notebook_name)[0])
            csv_writer.writerow([notebook_url, notebook_description])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--notebook-dir')
    parser.add_argument('-o', '--output-file')
    args = parser.parse_args()

    main(args.notebook_dir, args.output_file)
