import shlex
import subprocess
from pathlib import Path

import pagnn


def get_version():
    try:
        version = _git_commit_sha()
    except subprocess.CalledProcessError:
        version = pagnn.__version__
    return version


def _git_commit_sha():
    # Make sure all changes have been committed
    p = subprocess.run(
        shlex.split("git status -s"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=Path(__file__).absolute().parent.as_posix(),
        check=True,
    )
    if p.stdout.strip():
        raise Exception("All changes must be committed before running network!")
    # Get the hash of the current commit
    p = subprocess.run(
        shlex.split("git log -n1 --format='%h'"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=Path(__file__).absolute().parent.as_posix(),
        check=True,
    )
    return p.stdout.strip()
