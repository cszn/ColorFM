#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output-path", type=Path, required=True)

args = parser.parse_args()
args.output_path.mkdir(parents=True, exist_ok=True)

files = sorted(args.input_dir.iterdir(), key=lambda path: path.name)

copied = 0
for index, path in enumerate(files, start=1):
    if not path.is_file():
        continue
    output_path = args.output_path / f"{index:03d}.jpg"
    shutil.copy2(path, output_path)
    copied += 1

print(f"Copied {copied} renamed images to {args.output_path}")
