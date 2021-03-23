#!/usr/bin/env python3
from pathlib import Path

from tensorboard.main import run_main

def get_logdir(basedir: str):
    log_dirs = []
    for path in Path(basedir).absolute().glob('*/tensorboard/'):
        log_dirs.append(f'{path.parent.name}:{path}')
    return ','.join(log_dirs)


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, help='Base location for files')
    args = parser.parse_args()
    
    sys.argv = ['tensorboard', '--port', '6306', '--logdir', get_logdir(args.basedir)]
    sys.exit(run_main())

