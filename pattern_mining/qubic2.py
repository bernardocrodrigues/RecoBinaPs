""" quibic2.py

This module provides a wrapper around the QUBIC2 [1][2] biclustering algorithm public
implementation.

To avoid license issues, the QUBIC2 source code is not included in this repository. Instead, the
source code is fetched from the official GitHub repository and compiled locally.

After compilation, the QUBIC2 algorithm can be run on a given dataset.

[1] https://github.com/OSU-BMBL/QUBIC2
[2] Juan Xie, Anjun Ma, Yu Zhang, Bingqiang Liu, Sha Cao, Cankun Wang, Jennifer Xu, Chi Zhang,
    Qin Ma, QUBIC2: a novel and robust biclustering algorithm for analyses and interpretation of
    large-scale RNA-Seq data, Bioinformatics, Volume 36, Issue 4, February 2020, Pages 1143–1149,
    https://doi.org/10.1093/bioinformatics/btz692

Copyright 2022 Bernardo C. Rodrigues
See COPYING file for license details
"""

import os
import re
import logging
import subprocess
from pathlib import Path
import git
from pydantic import validate_call, ConfigDict

from typing import Annotated, Optional, List
from annotated_types import Gt, Le, Ge


from pattern_mining.formal_concept_analysis import create_concept, Concept
from . import DEFAULT_LOGGER

QUBIC2_GITHUB_URL = "https://github.com/OSU-BMBL/QUBIC2.git"
QUBIC2_GITHUB_SOURCE_REVISION = "5770b1d3c9e50ef41e98d9dfc2b1cb4f9ec9597a"
QUBIC2_DESTINATION_PATH = Path(os.environ.get("QUBIC2_DESTINATION_PATH", "/tmp/qubic2"))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def fetch_qubic2_source_code(destination_path: Path, logger: logging.Logger = DEFAULT_LOGGER):
    """
    Get the QUBIC2 source code from the official GitHub repository and clone it to the given
    destination path.

    Args:
        destination_path (Path): Path where the QUBIC2 source code will be cloned.
        logger (logging.Logger): Logger object to log messages.
    """

    logger.info(f"Cloning QUBIC2 source code from {QUBIC2_GITHUB_URL} to {destination_path}.")

    if destination_path.exists():
        raise ValueError(f"Destination path {destination_path} already exists.")
    destination_path.mkdir(parents=True)

    repo = git.Repo.init(destination_path)
    origin = repo.create_remote("origin", QUBIC2_GITHUB_URL)
    origin.fetch(QUBIC2_GITHUB_SOURCE_REVISION)
    origin.pull(QUBIC2_GITHUB_SOURCE_REVISION)
    repo.head.reset(QUBIC2_GITHUB_SOURCE_REVISION, index=True, working_tree=True)

    logger.info("Cloning OK")


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def compile_qubic2(source_path: Path, logger: logging.Logger = DEFAULT_LOGGER):
    """
    Compile the QUBIC2 source code in the given source path.

    Args:
        source_path (Path): Path where the QUBIC2 source code is located.
        logger (logging.Logger): Logger object to log messages.
    """

    cmd = ["make"]

    current_path = Path.cwd()
    os.chdir(source_path)

    logger.info(f"Compiling QUBIC2 source code in {source_path}.")

    output = subprocess.check_output(
        cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True
    )

    logger.debug(output)
    logger.info("Compilation OK")

    os.chdir(current_path)


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def is_qubic2_available(source_path: Path) -> bool:
    """
    Check if the QUBIC2 binary is available in the given source path.

    Args:
        source_path (Path): Path where the QUBIC2 source code is located.

    Returns:
        bool: True if the QUBIC2 binary is available, False otherwise.
    """
    return (source_path / "qubic").exists()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def run_qubic2(
    data_path: Path,
    bicluster_number: Annotated[int, Gt(0)] = 100,
    max_overlap: Annotated[float, Gt(0), Le(1)] = 1.0,
    consistency: Annotated[float, Gt(0.5), Le(1.0)] = 1.0,
    use_spearman_correlation: bool = False,
    minimum_column_width: Optional[Annotated[int, Ge(2)]] = None,
    logger: logging.Logger = DEFAULT_LOGGER,
):
    """
    Run the QUBIC2 algorithm on the given dataset.

    Args:
        data_path (Path): Path to the dataset.
        bicluster_number (int): Number of blocks to report.
        max_overlap (float): Maximum overlap between biclusters. Filters out biclusters with
                             overlap greater than this value.
        consistency (float): Consistency level of a block. The minimum ratio between the number of
                             identical valid symbols in a column and the total number of rows in the
                             output.
        use_spearman_correlation (bool): calculate the spearman correlation between any pair of
                                         genes this can capture more reliable relationship but much
                                         slower
        minimum_number_columns (int): Column width of the block. If not specified, the default
                                      value is 5% of the columns.
        logger (logging.Logger): Logger object to log messages.
    """

    cmd = [f"{QUBIC2_DESTINATION_PATH}/qubic"]
    cmd += ["-i", data_path.as_posix()]
    cmd += ["-o", str(bicluster_number)]
    cmd += ["-d"]
    if use_spearman_correlation:
        cmd += ["-p"]
    cmd += ["-f", str(max_overlap)]
    cmd += ["-c", str(consistency)]
    if minimum_column_width:
        cmd += ["-k", str(minimum_column_width)]

    logger.debug(f"Running QUBIC2 with command: {' '.join(cmd)}")

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)

    logger.debug(output)


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def parse_biclusters_from_qubic_output(result_path: str) -> List[Concept]:
    """
    Parse the biclusters from the QUBIC2 output file.

    Args:
        result_path (str): Path to the QUBIC2 output file.

    Returns:
        List[Concept]: List of formal concepts representing the biclusters.

    """
    regex_str = r"Genes \[\d+\]: (?P<extent>[\d ]+)\n Conds \[\d+\]: (?P<intent>[\d ]+)\n"
    regex = re.compile(regex_str, re.M)

    with open(result_path, "r") as f:
        output = f.read()

    matches = regex.findall(output)

    biclusters = []

    for match in matches:
        extent = [int(i) for i in match[0].split()]
        intent = [int(i) for i in match[1].split()]
        concept = create_concept(extent, intent)
        biclusters.append(concept)

    return biclusters
