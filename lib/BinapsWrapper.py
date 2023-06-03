# Copyright 2022 Bernardo C. Rodrigues
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
# have received a copy of the GNU General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

"""
utils.py

Series of helper functions to make interacting with BinaPs vanilla implementation more friendly.
"""

import re
from subprocess import run
from tabulate import tabulate
from typing import List, Optional, TextIO


def generate_synthetic_data(row_quantity, column_quantity, file_prefix: str, max_pattern_size: int,
                            noise: float = 0.5, density: float = 0.5) -> None:
    """Generates a synthetic database based on known patterns.

    This is a wrapper function over the vanilla R script generate_toy.R. It places 4 files at the
    current work directory:
        <file_prefix>.dat - row_quantity x column_quantity dataset
        <file_prefix>.dat_patterns.txt - real patterns in <file_prefix>.dat
        <file_prefix>_itemOverlap.dat row_quantity x column_quantity dataset
        <file_prefix>_itemOverlap.dat_patterns.txt - real patterns in <file_prefix>_itemOverlap.dat

    Arbitrary noise is added to test the BinaPS robustness. In <file_prefix>_itemOverlap.dat the
    patterns may overlap (e.g ABC CDE CEF) while in <file_prefix>.dat they may not (e.g. AB C DE F).

    Args:
        row_quantity: number of rows, or transactions, that the synthetic dataset should have
        column_quantity: number of columns that the synthetic dataset should have. In other words,
            how many possible atributes, or items, each row may have.
        file_prefix: prefix that should be present on all generated files.
        max_pattern_size: biggest pattern size allowed to be used generated
        noise: percentage of the dataset to be flipped randomly
        density: percentage of the dataset that is non-zero
    """

    assert max_pattern_size > 0
    assert 0 <= noise <= 1
    assert 0 <= density <= 1

    cmd = f"Rscript binaps/Data/Synthetic_data/generate_toy.R AND {column_quantity} " \
          f"{row_quantity} {max_pattern_size} {file_prefix} {noise} {density}"

    print(cmd)
    output = run(cmd.split(" "), capture_output=True, check=True)
    print(output.stdout.decode())


def run_binaps(data_path: str, hidden_dimension: int, epochs: int) -> None:
    """ Runs BinaPs Autoencoder

    This will ingest a given dataset and extract its patterns through BinaPs. This will place a
    .binaps.patterns file on the current working dir with the pattern list

    Args:
        data_path: path for a .dat format database. generate_synthetic_data provides data in the
            expected format, use it as template.
        hidden_dimension: number of neurons in the hidden dimension. This also indicates the number
            of inferred patterns. Each neuron will correspond to a single pattern.
        epochs: how many epochs will the neural network train
    """

    cmd = f"python3 binaps/Binaps_code/main.py -i {data_path} --hidden_dim " \
          f"{hidden_dimension} --epochs={epochs}"

    print(cmd)
    output = run(cmd.split(" "), capture_output=True, check=True)
    stdout = output.stdout.decode()

    print_character_length = 300
    print(f"{stdout[:print_character_length]} [output truncated]")
    print("...")
    print(f"[output truncated] {stdout[-print_character_length:]}")


def parse_binaps_patterns(file_object: TextIO) -> List[List[int]]:
    """
    Parse binaps patterns from a file.

    Args:
        file_object: A file-like object containing the binaps detected patterns.

    Returns:
        A list of lists representing the parsed binaps patterns.

    This function reads the contents of the provided file and extracts binaps patterns.
    Each binaps pattern is represented as a list of integers. The function returns a list
    of these patterns.

    Example:
        patterns_file = open('patterns.txt', 'r')
        patterns = parse_binaps_patterns(patterns_file)
        patterns_file.close()
    """

    file_contents = file_object.read()

    regex = re.compile(r"\[[ ]*([ \d\n]+)]")
    patterns_as_strings = regex.findall(file_contents)

    patterns = []
    for pattern_as_string in patterns_as_strings:
        pattern = [int(d) for d in pattern_as_string.replace('\n', ' ').split()]
        patterns.append(pattern)

    return patterns


def compare_datasets_based_on_f1(estimated_patterns_file: str, real_patterns_file: str) -> None:
    """Get the F1 Score on the inferred dataset.

    Args:
        estimated_patterns_file: path for a BinaPs' .binaps.patterns output file
        real_patterns_file: path for a .dat_patterns.txt patterns file as generated by
            generate_synthetic_data.
    """

    cmd = f"python3 binaps/Data/Synthetic_data/comp_patterns.py -p {estimated_patterns_file} " \
          f"-t Binaps -r {real_patterns_file} -m F1"

    output = run(cmd.split(" "), capture_output=True, check=True)
    print(output.stdout.decode())


def display_as_table(data: List[List], headers: Optional[List[str]] = [], title: Optional[str] = None) -> None:
    """
    Display data as a formatted table.

    Args:
        data: A list of lists representing the data to be displayed.
        headers (optional): A list of strings representing the column headers (default: []).
        title (optional): A string representing the title of the table (default: None).

    Returns:
        None

    This function takes the provided data and displays it as a formatted table. The data should be
    provided as a list of lists, where each inner list represents a row of the table. The column headers,
    if provided, should be specified as a list of strings. If a title is provided, it will be displayed
    above the table.

    Example:
        data = [
            ['John Doe', 25, 'Engineer'],
            ['Jane Smith', 32, 'Manager'],
            ['Mark Johnson', 41, 'Developer']
        ]

        headers = ['Name', 'Age', 'Role']

        display_as_table(data, headers=headers, title='Employee Information')

        # Output:
        # Employee Information
        # ╒═════════════╤═════╤═════════════╕
        # │ Name        │ Age │ Role        │
        # ╞═════════════╪═════╪═════════════╡
        # │ John Doe    │ 25  │ Engineer    │
        # ├─────────────┼─────┼─────────────┤
        # │ Jane Smith  │ 32  │ Manager     │
        # ├─────────────┼─────┼─────────────┤
        # │ Mark Johnson│ 41  │ Developer   │
        # ╘═════════════╧═════╧═════════════╛
    """
    if title:
        print(title)
    print(tabulate(data, headers = headers, tablefmt="fancy_grid"))