# Copyright 2022 Bernardo C. Rodrigues
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
# have received a copy of the GNU General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# This module comprises all methods and classes related to Formal Concept Analysis.
#
# Bibliography
# [1] Discovery of optimal factors in binary data via a novel method of matrix decomposition <https://www.sciencedirect.com/science/article/pii/S0022000009000415>


import numpy as np
from numba import njit
from numba.typed import List
from lib.BinaryDataset import BinaryDataset

from collections import namedtuple

Concept = namedtuple("Concept", "extent intent")


@njit
def submatrix_intersection_size(rows, columns, U) -> int:  # pragma: no cover
    intersection_size = 0
    for row in rows:
        for column in columns:
            if U[row][column]:
                intersection_size += 1
    return intersection_size


@njit
def erase_submatrix_values(rows, columns, U) -> None:  # pragma: no cover
    for row in rows:
        for column in columns:
            U[row][column] = False


def GreConD(binary_dataset: BinaryDataset, coverage=1, verbose=False):
    """
    Implements Algorithm 2 in section 2.5.2 (page 15) from [1].

    This algorithms proposes a greedy heuristic to enumerate the set F given a binary dataset D. F is suposed to be a
    'good enough' formal context of D although it's not guaranteed to be optimal (smallest F that covers all of D).

    It is also possible to define the desired coverage. The algorithm will stop when the current set F covers the given
    coverage.
    """

    if verbose:
        print("[GreConD] Mining Formal Concepts...")

    U = binary_dataset.get_raw_copy()
    initial_number_of_trues = np.count_nonzero(U)

    if verbose:
        print(f"[GreConD] Binary dataset has {np.count_nonzero(U)} True's (sparcity: {initial_number_of_trues/U.size:.2f})")

    F = []
    current_coverage = 0

    while coverage > current_coverage:
        current_coverage = 1 - np.count_nonzero(U) / initial_number_of_trues
        D = np.array([])
        V = 0
        D_u_j = np.array([])  # current D union {j}

        searching = True
        js_not_in_D = [j for j in binary_dataset.Y if j not in D_u_j]
        
        while searching:

            best_D_u_j_closed_intent = None
            best_D_u_j_V = 0

            for j in js_not_in_D:

                D_u_j = np.append(D, j).astype(int)

                D_u_j_closed_extent = binary_dataset.t(D_u_j)
                D_u_j_closed_intent = binary_dataset.i(D_u_j_closed_extent)

                D_u_j_V = submatrix_intersection_size(D_u_j_closed_extent, D_u_j_closed_intent, U)

                if D_u_j_V > best_D_u_j_V:
                    best_D_u_j_V = D_u_j_V
                    best_D_u_j_closed_intent = D_u_j_closed_intent.copy()

            if best_D_u_j_V > V:
                D = best_D_u_j_closed_intent
                V = best_D_u_j_V
            else:
                searching = False

        C = binary_dataset.t(D)

        new_concept = Concept(C, D)

        F.append(new_concept)

        erase_submatrix_values(new_concept.extent, new_concept.intent, U)

        current_coverage = 1 - np.count_nonzero(U) / initial_number_of_trues

        if verbose:
            print(f"[GreConD] Current Coverage: {current_coverage*100:.2f}%", end="\r")

    if verbose:
        print("[GreConD] Mining Formal Concepts OK")
        print(f"[GreConD] Formal Concepts mined: {len(F)}")
        print(f"[GreConD] Final Concepts Coverage {current_coverage*100}%")

    return F, current_coverage


@njit
def _get_matrices(concepts: List, dataset_number_rows, dataset_number_cols):  # pragma: no cover

    Af = []
    Bf = []

    for concept in concepts:

        column = [0] * dataset_number_rows
        row = [0] * dataset_number_cols

        for item in concept.extent:
            column[item] = 1

        for item in concept.intent:
            row[item] = 1

        Af.append(column)
        Bf.append(row)

    return Af, Bf


def get_factor_matrices_from_concepts(concepts, dataset_number_rows, dataset_number_cols):
    """
    Given a list of concepts, this method will return two matrices as described in section 2.1 (page 6) from [1].

    If the given formal concepts cover all values from a matrix I, I = Af x Bf.
    """

    typed_a = List()
    [typed_a.append(x) for x in concepts]

    Af, Bf = _get_matrices(typed_a, dataset_number_rows, dataset_number_cols)

    Af = np.array(Af, dtype=bool).T
    Bf = np.array(Bf, dtype=bool)

    return Af, Bf
