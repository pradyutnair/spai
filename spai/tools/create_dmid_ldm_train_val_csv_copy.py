#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script for generating a CSV file for LSUN images from train, val, or directly from LSUN directory."""
from pathlib import Path
import random
from typing import Any, Optional
import os
import click
from tqdm import tqdm
import sys

# Set the path to the spai directory
print("Setting the path to the spai directory")
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(base)
sys.path.append(base)

from spai import data_utils

__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.1"
__revision__: int = 2

@click.command()
@click.option("--train_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Directory containing real_lsun.txt for training split")
@click.option("--val_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Directory containing real_lsun.txt for validation split")
@click.option("--lsun_dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory with LSUN LMDB zip files")
@click.option("--real_lsun_filename", type=str, default="real_lsun.txt",
              help="Filename listing real LSUN identifiers within train/val dirs")
@click.option("-o", "--output_csv",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True,
              help="Path to write the resulting CSV")
@click.option("-r", "--csv_root_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Root directory for relative paths in CSV")
@click.option("-d", "--output_csv_delimiter", type=str, default=",", show_default=True,
              help="Delimiter to use in the output CSV")
@click.option("-n", "--samples_num", type=int, default=None, show_default=True,
              help="Optionally sample a subset of entries")
@click.option("-f", "--filter", type=str, multiple=True,
              help="Optionally filter by LSUN category (e.g. bedroom)")
def main(
    train_dir: Optional[Path],
    val_dir: Optional[Path],
    lsun_dir: Path,
    real_lsun_filename: str,
    output_csv: Path,
    csv_root_dir: Optional[Path],
    output_csv_delimiter: str,
    samples_num: Optional[int],
    filter: list[str]
) -> None:
    # Determine CSV root for relative paths
    if csv_root_dir is None:
        csv_root_dir = output_csv.parent

    entries: list[dict[str, Any]] = []

    def rel_path(p: Path) -> Path:
        try:
            return p.relative_to(csv_root_dir)
        except ValueError:
            return p

    # Collect splits
    split_dirs = []
    split_labels = []
    if train_dir is not None:
        split_dirs.append(train_dir);   split_labels.append("train")
    if val_dir   is not None:
        split_dirs.append(val_dir);     split_labels.append("val")

    # If train/val provided, use real_lsun.txt to pick zips
    if split_dirs:
        for s_dir, s_label in tqdm(zip(split_dirs, split_labels),
                                   desc="Finding LSUN samples in splits", unit="image"):
            real_lsun_file = s_dir / real_lsun_filename
            samples = find_lsun_samples(real_lsun_file, lsun_dir, s_label)
            for p in samples:
                cat = p.name.split("_")[0]
                if filter and cat not in filter:
                    continue
                entries.append({"image": str(rel_path(p)), "class": 0, "split": s_label})
    else:
        # No split dirs: auto-scan all .zip in lsun_dir
        for p in tqdm(lsun_dir.glob("*.zip"), desc="Scanning LSUN directory", unit="file"):
            name = p.name
            if name.endswith("_train_lmdb.zip"):
                split = "train"
            elif name.endswith("_val_lmdb.zip"):
                split = "val"
            elif name == "test_lmdb.zip":
                split = "test"
            else:
                continue
            cat = name.split("_")[0]
            if filter and cat not in filter:
                continue
            entries.append({"image": str(rel_path(p)), "class": 0, "split": split})

    # Optionally subsample
    if samples_num is not None and entries:
        entries = random.sample(entries, samples_num)

    # Write out CSV
    if not entries:
        print("No entries to write to CSV.")
        return

    data_utils.write_csv_file(entries, output_csv, delimiter=output_csv_delimiter)
    print(f"Exported CSV to {output_csv}")


def find_lsun_samples(lsun_real_file: Path, cnndetect_dir: Path, split: str) -> list[Path]:
    assert split in ["train", "val"], "Split must be 'train' or 'val'"
    print("Loading LSUN image paths from list.")
    with lsun_real_file.open() as f:
        lines = [l.rstrip() for l in f]
    files: list[Path] = []
    for l in lines:
        parts = l.split("_")
        cat = parts[0]
        zip_file = cnndetect_dir / f"{cat}_{split}_lmdb.zip"
        if zip_file.exists():
            files.append(zip_file)
    print("Completed loading LSUN image paths.")
    for f in tqdm(files, desc="Verifying LSUN files", unit="file"):
        assert f.exists()
    return files


def read_lmdb_file(lmdb_path: Path) -> list[bytes]:
    """Read data from an LMDB file."""
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        data = [value for key, value in cursor]
    env.close()
    return data


if __name__ == "__main__":
    print("Starting script")
    lsun_dir = Path("/scratch-shared/dl2_spai_datasets/LSUN/lsun/scenes/")
    output_csv = Path("/home/pnair/spai/datasets/lsun_train_val_lsun_test.csv")
    main(lsun_dir=lsun_dir, output_csv=output_csv, train_dir=None, val_dir=None, real_lsun_filename="real_lsun.txt", csv_root_dir=None, output_csv_delimiter=",", samples_num=None, filter=[])
