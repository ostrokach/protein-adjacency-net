#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).absolute().parent.parent

if __name__ == "__main__":
    stage = sys.argv[-1] if len(sys.argv) > 1 else None

    with PROJECT_ROOT.joinpath(".gitlab-ci.yml").open("rt") as fin:
        ci_data = yaml.load(fin)

    if stage is None:
        variables = {**ci_data.get("variables", {})}
    else:
        if stage not in ci_data:
            raise Exception(f"No stage with name '{stage}'!")
        variables = {**ci_data.get("variables", {}), **ci_data[sys.argv[1]].get("variables", {})}

    variables = {
        k: (
            v
            if "$" not in v
            else subprocess.check_output(["bash", "-c", f'echo "{v}"']).decode().strip()
        )
        for k, v in variables.items()
    }

    # Get a list of current environments

    proc = subprocess.run(
        ["conda", "env", "list"], stdout=subprocess.PIPE, universal_newlines=True, check=True
    )
    envs = [line.split()[0] for line in proc.stdout.strip().split("\n")]

    # Print output for sourcing a bash script

    print("set -ev")

    print("# Export environment variables")
    for key, value in variables.items():
        print(f"export {key}='{value}'")
    print()

    if variables["CONDA_ENVIRONMENT_NAME"] not in envs:
        print("# Create conda environment")
        print(f"conda env create -n {variables['CONDA_ENVIRONMENT_NAME']} -f environment-lock.yaml")
        for nbextension in [
            "collapsible_headings/main",
            "runtools/main",
            "codefolding/main",
            "code_prettify/isort",
            "toc2/main",
        ]:
            print(f"jupyter nbextension enable {nbextension}")
    print()

    print("# Activate conda environment")
    print(f"source activate {variables['CONDA_ENVIRONMENT_NAME']}\n")
