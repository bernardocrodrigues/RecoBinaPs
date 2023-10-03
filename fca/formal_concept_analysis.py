""" formal_concept_analysis.py

This module implements the GreConD algorithm [1] for mining formal concepts from a binary dataset.
It also implements some helper functions to aid in the manipulation of formal concepts.

Copyright 2022 Bernardo C. Rodrigues
See COPYING file for license details

Bibliography
[1] Discovery of optimal factors in binary data via a novel method of matrix decomposition 
    <https://www.sciencedirect.com/science/article/pii/S0022000009000415>
"""

# We shall disable the invalid-name warning for this file because we are using the same variable
# names as in the original paper [1] to make it easier to understand the code.
# pylint: disable=C0103

from collections import namedtuple

import numpy as np
import numba as nb

from typing import List

from dataset.binary_dataset import BinaryDataset, _it

from . import DEFAULTLOGGER

Concept = namedtuple("Concept", "extent intent")


# @nb.njit
def grecond(binary_dataset: np.array, coverage=1) -> List[Concept]:  # pragma: no cover
    """
    Implements Algorithm 2 in section 2.5.2 (page 15) from [1].

    This algorithms proposes a greedy heuristic to enumerate the set F given a binary dataset D.
    F is supposed to be a 'good enough' formal context of D although it's not guaranteed to be
    optimal (smallest F that covers all of D).

    It is also possible to define the desired coverage. The algorithm will stop when the current set
    F covers the given coverage.
    """

    def get_submatrix_intersection_size(rows: List[int], columns: List[int], U: np.array) -> int:
        intersection_size = 0
        for row in rows:
            for column in columns:
                if U[row][column]:
                    intersection_size += 1
        return intersection_size

    def erase_submatrix_values(
        rows: List[int], columns: List[int], U: np.array
    ) -> None:  # pragma: no cover # pylint: disable=invalid-name
        for row in rows:
            for column in columns:
                U[row][column] = False

    U = binary_dataset.copy()
    transposed = binary_dataset.T
    initial_number_of_trues = np.count_nonzero(U)

    Y = np.arange(binary_dataset.shape[1])  # pylint: disable=invalid-name

    F = nb.typed.List()
    current_coverage = 0

    while coverage > current_coverage:
        D = nb.typed.List.empty_list(nb.types.int64)  # analogous to D = []
        V = 0
        D_u_j = nb.typed.List.empty_list(nb.types.int64)  # analogous to D_u_j = []

        searching = True
        js_not_in_D = [j for j in Y if j not in D_u_j]

        while searching:
            best_D_u_j_closed_intent = nb.typed.List.empty_list(nb.types.int64)
            best_D_u_j_V = 0

            for j in js_not_in_D:
                D_u_j.append(j)

                D_u_j_closed_extent = _it(binary_dataset, D_u_j)
                D_u_j_closed_intent = _it(transposed, D_u_j_closed_extent)

                D_u_j_V = get_submatrix_intersection_size(
                    D_u_j_closed_extent, D_u_j_closed_intent, U
                )

                if D_u_j_V > best_D_u_j_V:
                    best_D_u_j_V = D_u_j_V
                    best_D_u_j_closed_intent = D_u_j_closed_intent.copy()
                else:
                    D_u_j.remove(j)

            if best_D_u_j_V > V:
                D = best_D_u_j_closed_intent
                V = best_D_u_j_V
            else:
                searching = False

        C = _it(binary_dataset, D)

        new_concept = Concept(extent=C, intent=D)

        F.append(new_concept)

        erase_submatrix_values(C, D, U)

        current_coverage = 1 - np.count_nonzero(U) / initial_number_of_trues

    return F, current_coverage


@nb.njit
def _get_matrices(
    concepts: List[Concept], dataset_number_rows: int, dataset_number_cols: int
):  # pragma: no cover
    """
    This function acts as a kernel for the get_factor_matrices_from_concepts function. It is
    implemented in Numba to speed up the process of creating the matrices. It is not meant to be
    called directly. Use the get_factor_matrices_from_concepts function instead.
    """
    Af = np.zeros(shape=(dataset_number_rows, len(concepts)), dtype=nb.bool_)
    Bf = np.zeros(shape=(len(concepts), dataset_number_cols), dtype=nb.bool_)

    for concept_index, concept in enumerate(concepts):
        for item in concept.extent:
            Af[item][concept_index] = True

        for item in concept.intent:
            Bf[concept_index][item] = True

    return Af, Bf


def get_factor_matrices_from_concepts(
    concepts: List[Concept], dataset_number_rows: int, dataset_number_cols: int
):
    """
    This function takes a list of formal concepts and returns two matrices as described in section
    2.1 (page 6) from [1]. If the given formal concepts cover all values from a matrix I,
    I = Af x Bf.

    Args:
        concepts: A list of formal concepts.
        dataset_number_rows: The number of rows in the dataset.
        dataset_number_cols: The number of columns in the dataset.

    Returns:
        Af: A matrix with the same number of rows as the dataset and the same number of columns as
            the number of concepts. Each column represents a formal concept and each row represents
            an object in the dataset. If an object belongs to a concept, the value in the
            corresponding cell will be 1, otherwise it will be 0.
        Bf: A matrix with the same number of rows as the number of concepts and the same number of
            columns as the dataset. Each row represents a formal concept and each column represents
            an attribute in the dataset. If an attribute belongs to a concept, the value in the
            corresponding cell will be 1, otherwise it will be 0.


    Example:
        concepts = [Concept([0, 1], [0, 1]), Concept([0, 1, 2], [0, 1, 2])]
        dataset_number_rows = 4
        dataset_number_cols = 4
        Af, Bf = get_factor_matrices_from_concepts(concepts, dataset_number_rows,
                                                   dataset_number_cols)
        Af  # returns np.array([[1, 1], [1, 1], [0, 1], [0, 1]])
        Bf  # returns np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
    """

    # We need to convert the list of concepts to a Numba typed list to be able to use it in the
    # _get_matrices function.
    typed_concept_list = nb.typed.List()
    for concept in concepts:
        typed_concept_list.append(concept)

    return _get_matrices(typed_concept_list, dataset_number_rows, dataset_number_cols)


def construct_context_from_binaps_patterns(
    binary_dataset: BinaryDataset, patterns: List[List[int]], closed_itemsets: bool = True
) -> List[Concept]:
    """
    Construct a context from binaps patterns.

    Args:
        binary_dataset: The binary dataset object.
        patterns: A list of binaps patterns represented as lists of integers.
        closed_itemsets: A boolean flag indicating whether to compute closed itemsets
                         (default: True).

    Returns:
        A list of Concept objects representing the constructed context.

    This function constructs a context from the given binaps patterns and the associated binary
    dataset. Each binaps pattern is converted into a tidset and itemset based on the binary dataset.
    The context is represented as a list of Concept objects, where each Concept consists of a tidset
    and an itemset.

    If the `closed_itemsets` flag is set to True, closed itemsets will be computed by transforming
    the itemset into a closed itemset based on the binary dataset.

    Example:
        binary_dataset = BinaryDataset(...)
        patterns = [[1, 2, 3], [4, 5], [2, 4, 6]]
        context = construct_context_from_binaps_patterns(binary_dataset, patterns)
    """
    context = []

    for pattern in patterns:
        tidset = binary_dataset.t(pattern)  # a pattern equals an itemset

        if closed_itemsets:
            itemset = binary_dataset.i(tidset)
            tidset = binary_dataset.t(itemset)
        else:
            itemset = pattern

        context.append(Concept(tidset, itemset))

    return context


# # Numba Compilation
# # Numba uses a Just-In-Time compiler to speed up the execution of the code. The functions need to
# # be ran once to be compiled. Therefore, we run the functions at import time to avoid the overhead
# # of compiling the functions when they are called.
# get_factor_matrices_from_concepts([Concept([0, 1], [0, 1]), Concept([0, 1, 2], [0, 1, 2])], 4, 4)
# erase_submatrix_values(
#     [0, 1], [0, 2], np.array([[True, False, True], [False, True, True], [True, True, True]])
# )
# submatrix_intersection_size(
#     [0, 1], [0, 2], np.array([[True, False, True], [False, True, True], [True, True, True]])
# )
