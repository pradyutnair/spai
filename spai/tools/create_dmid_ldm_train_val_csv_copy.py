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

"""Script for generating a CSV file for LSUN images from LMDB files."""
from pathlib import Path
import random
from typing import Any, Optional
import os
import click
from tqdm import tqdm
import sys
import lmdb
import numpy as np
import shutil
import zipfile
import cv2

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

def extract_lmdb(zip_path: Path, extract_dir: Path) -> Path:
    """Extract LMDB file from zip archive to permanent directory."""
    # Create category directory
    category = zip_path.stem.split('_')[0]
    category_dir = extract_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract to category directory
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(category_dir)
    except zipfile.BadZipFile:
        print(f"Warning: {zip_path} is not a valid zip file, trying to use it directly as LMDB")
        # Copy the file directly if it's not a zip
        shutil.copy2(zip_path, category_dir / zip_path.name)
    
    return category_dir / zip_path.stem

def extract_images_from_lmdb(lmdb_path: Path, output_dir: Path) -> list[str]:
    """Extract images from LMDB file and save them to output directory."""
    env = lmdb.open(str(lmdb_path), map_size=1099511627776,
                   max_readers=100, readonly=True)
    image_paths = []
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            # Decode key to get image path
            img_key = key.decode('ascii')
            # Create output path
            img_path = output_dir / f"{img_key}.webp"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Decode and save image
            img = cv2.imdecode(np.frombuffer(val, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(img_path), img)
            image_paths.append(str(img_path))
    
    env.close()
    return image_paths

@click.command()
@click.option("--lsun_dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory with LSUN LMDB zip files")
@click.option("--extract_dir", required=True,
              type=click.Path(exists=False, file_okay=False, path_type=Path),
              help="Directory to extract LMDB files to")
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
    lsun_dir: Path,
    extract_dir: Path,
    output_csv: Path,
    csv_root_dir: Optional[Path],
    output_csv_delimiter: str,
    samples_num: Optional[int],
    filter: list[str]
) -> None:
    # Create extract directory if it doesn't exist
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine CSV root for relative paths
    if csv_root_dir is None:
        csv_root_dir = output_csv.parent

    entries: list[dict[str, Any]] = []

    def rel_path(p: Path) -> Path:
        try:
            return p.relative_to(csv_root_dir)
        except ValueError:
            return p

    # Show the lsun_dir
    print(f"LSUN directory: {lsun_dir}")
    print(f"Extract directory: {extract_dir}")
    print(f"Contents of LSUN directory: {os.listdir(lsun_dir)}")

    # Process each zip file
    for zip_file in tqdm(list(lsun_dir.glob("*.zip")), desc="Processing LSUN files"):
        print(f"Processing {zip_file}")
        
        # Determine split from filename
        if "_train_lmdb" in zip_file.name:
            split = "train"
        elif "_val_lmdb" in zip_file.name:
            split = "val"
        elif zip_file.name == "test_lmdb.zip":
            split = "test"
        else:
            continue

        try:
            # Extract LMDB file
            lmdb_path = extract_lmdb(zip_file, extract_dir)
            print(f"Extracted LMDB to {lmdb_path}")
            
            # Extract images from LMDB
            image_paths = extract_images_from_lmdb(lmdb_path, lmdb_path)
            print(f"Extracted {len(image_paths)} images from {zip_file.name}")
            
            # Add entries to CSV
            for img_path in image_paths:
                entries.append({
                    "image": str(rel_path(Path(img_path))),
                    "class": 0,
                    "split": split
                })
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")
            continue

    # Optionally subsample
    if samples_num is not None and entries:
        entries = random.sample(entries, samples_num)

    # Write out CSV
    if not entries:
        print("No entries to write to CSV.")
        return

    data_utils.write_csv_file(entries, output_csv, delimiter=output_csv_delimiter)
    print(f"Exported CSV to {output_csv}")

if __name__ == "__main__":
    print("Starting script")
    lsun_dir = Path("/scratch-shared/dl2_spai_datasets/LSUN/lsun/scenes/")
    extract_dir = Path("/scratch-shared/dl2_spai_datasets/LSUN/lsun/scenes/extracted")
    output_csv = Path("/home/pnair/spai/datasets/lsun_train_val_lsun_test.csv")
    main.callback(lsun_dir=lsun_dir, extract_dir=extract_dir, output_csv=output_csv, 
                 csv_root_dir=None, output_csv_delimiter=",", samples_num=None, filter=[])
