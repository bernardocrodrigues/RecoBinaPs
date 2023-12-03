#!/usr/bin/env python3
""" generate_movielens_folds.py

Given the MovieLens 100K dataset, which includes pre-defined folds, this script generates the
folds in the format required by BinaPs. The MovieLens 100K dataset can be downloaded from
https://grouplens.org/datasets/movielens/100k/. The folds are generated in the same directory
where the dataset is located. 

Folds nomenclature follows the MovieLens 100K dataset, for example, u1.base will be transformed
into u1.base.dat and so on.

Test folds are not generated.

Usage:
    The preferred way to run this script withing the accompanying project container. To do so,
    run the following command from the project environment directory:

    docker-compose run --rm --entrypoint="scripts/generate_movielens_folds.py" notebook-cuda <output_dir>

Copyright 2023 Bernardo C. Rodrigues
See COPYING file for license details
"""

import argparse
from pathlib import Path

from surprise import Dataset, Reader
from surprise.model_selection import PredefinedKFold

from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
    save_as_binaps_compatible_input,
)
from dataset.movie_lens import download_movielens

MOVIELENS_100K_DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_URLS = {"100k": MOVIELENS_100K_DOWNLOAD_URL, "1m": MOVIELENS_1M_DOWNLOAD_URL}


def convert_to_binaps_compatible_folds(movielens_path: Path):
    """
    Generate the folds in the format required by BinaPs. Given the MovieLens 100K dataset, which
    includes pre-defined folds, this script generates the folds in the format required by BinaPs.

    Folds nomenclature follows the MovieLens 100K dataset, for example, u1.base will be transformed
    into u1.base.dat and so on.

    Test folds are not generated.


    Args:
        movielens_path: The path to the directory where the MovieLens 100K dataset is located.

    """
    reader = Reader("ml-100k")

    folds_files = [
        (movielens_path / f"u{i}.base", movielens_path / f"u{i}.test") for i in (1, 2, 3, 4, 5)
    ]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()

    for index, (trainset, _) in enumerate(pkf.split(data)):
        binary_dataset = load_binary_dataset_from_trainset(trainset)

        with open(movielens_path / f"u{index}.base.dat", "w", encoding="UTF-8") as file_object:
            save_as_binaps_compatible_input(binary_dataset, file_object)


def get_parser() -> argparse.ArgumentParser:
    """
    Get the command line parser for the script. It defines the command line arguments and their
    types.

    Returns:
        The command line parser for the script.
    """
    parser = argparse.ArgumentParser(
        description="Given the MovieLens 100K dataset, which includes pre-defined folds, this "
        "script generates the folds in the format required by BinaPs."
    )
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    return parser


def main():
    """
    Main function. It parses the command line arguments and calls the functions to download the
    MovieLens 100K dataset and generate the folds in the format required by BinaPs.
    """
    parser = get_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    download_movielens(output_dir)
    convert_to_binaps_compatible_folds(output_dir / "ml-100k")


if __name__ == "__main__":
    main()
