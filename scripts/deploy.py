#!/usr/bin/env python3
import logging
import os
import os.path as op
import shlex
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(op.dirname(op.dirname(op.abspath(__file__))))


def copy_files(src, dest):
    system_command = f"rsync -av {src}/ {dest}/"
    logger.info("Running system command: '%s'", system_command)
    subprocess.run(shlex.split(system_command), check=True)


def main():
    with PROJECT_ROOT.joinpath(".gitlab-ci.yml").open() as fin:
        data = yaml.load(fin)
    name = PROJECT_ROOT.name
    version = data["variables"]["PACKAGE_VERSION"]
    paths = data["deploy"]["artifacts"]["paths"]
    for path in paths:
        src = PROJECT_ROOT.joinpath(path)
        dest = f"{os.environ['DATABIN_DIR']}/{name}/{version}/{src.name}"
        copy_files(src, dest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
