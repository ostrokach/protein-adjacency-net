#!/usr/bin/env python3.6
import argparse
import shutil
from pathlib import Path

import yaml


def allow_deletion():
    message = "This will remove all files in the figures directory. Continue (y/n)? "
    prompt = input(message)
    while prompt not in ["y", "n"]:
        prompt = input(message)
    return prompt == "y"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    with Path(__file__).with_name("figures.yml").open("rt") as fin:
        data = yaml.load(fin)
    if args.delete and not allow_deletion():
        return
    for folder, files in data.items():
        if not files:
            continue
        for figure in files:
            src_path = Path(folder).joinpath(figure).expanduser()
            dest_path = Path(__file__).absolute().with_name(src_path.name)
            # Copy PDFs
            print(f"'{src_path}' -> '{dest_path}'")
            shutil.copy(src_path, dest_path)
            # Copy PNGs
            if src_path.suffix == ".pdf" and src_path.with_suffix(".png").is_file():
                print(f"'{src_path.with_suffix('.png')}' -> '{dest_path.with_suffix('.png')}'")
                shutil.copy(src_path.with_suffix(".png"), dest_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
