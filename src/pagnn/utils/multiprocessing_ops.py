import logging

import psutil

logger = logging.getLogger(__name__)


def kill_tree(pid, including_parent=False):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        logger.warning("Killing child process: %s", child)
        child.kill()

    if including_parent:
        parent.kill()
