#!/usr/bin/env python
import time
from pathlib import Path

import nbformat
import zmq
from nbconvert import html
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


def main(input_file, output_file):
    """Execute a Jupyter notebook and save the results as HTML.

    Args:
        input_file: Location of the notebook to execute.
        output_file: Location where to write the output log file.
    """
    ep = ExecutePreprocessor(timeout=60 * 60 * 24 * 7)  # one week

    with open(input_file) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    notebook_path = Path(__file__).resolve().parent.parent.joinpath("notebooks/").as_posix()

    try:
        nb, _ = ep.preprocess(nb, {"metadata": {"path": notebook_path}})
    except CellExecutionError:
        print(f"Error executing notebook '{input_file}'; see '{output_file}' for traceback.")
        raise
    finally:
        exporter = html.HTMLExporter(template_file="full")
        output, _ = exporter.from_notebook_node(nb)
        with open(output_file, mode="wt") as fout:
            fout.write(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file")
    parser.add_argument("-o", "--output-file")
    args = parser.parse_args()

    n_tries = 0
    while True:
        try:
            main(args.input_file, args.output_file)
            break
        except (zmq.error.ZMQError, RuntimeError) as e:
            print(f"Caught an error: '{e}'.", end=" ")
            n_tries += 1
            if n_tries > 5:
                print("Giving up!")
                raise
            else:
                print("Retrying...")
                time.sleep(10 * n_tries)
                continue
