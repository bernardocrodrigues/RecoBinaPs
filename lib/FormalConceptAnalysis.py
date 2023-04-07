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

def  GreConD(binary_dataset: BinaryDataset, coverage=1, verbose=False):
    """
    Wrapper over Algorithm2 to match my API
    """

    F = Algorithm2(binary_dataset._binary_dataset, coverage)

    F_as_concepts = []
    for item in F:
        F_as_concepts.append(Concept(item[0], item[1]))

    return F_as_concepts, coverage

# Parameters:
# I: Boolean matrix
# mincov: minimum coverage [0,1]
def Algorithm2(I, mincov):
    U = (I.copy()).ravel()
    F = []
    nt1s = U.sum()
    while 1 - (U.sum() / nt1s) < mincov:
        D = []
        V = 0
        existe = True
        while existe:
            existe = False
            bestDbolaMaisJ = 0
            bestDjClosed = []
            for j in range(I.shape[1]):
                if not j in D:
                    Dj = D.copy()
                    Dj.append(j)
                    nIntersection, DjClosed = nbolaMais(Dj, U, I)
                    if nIntersection > V:
                        existe = True
                        if nIntersection > bestDbolaMaisJ:
                            bestDbolaMaisJ = nIntersection #using idempotency property
                            bestDjClosed = DjClosed
        D = bestDjClosed
        V = bestDbolaMaisJ
        if existe:
            C = downArrow(D, I)
            F.append([C,D])
            idx = np.ravel_multi_index([np.repeat(C, len(D)),np.tile(D,len(C))], I.shape)
            U[idx] = 0
    return F

def nbolaMais(Dy, U, matrix):
    rows = downArrow(Dy, matrix)
    if len(rows) == 0:
        return 0, []
    cols = upArrow(rows, matrix)
    idx = np.ravel_multi_index([np.repeat(rows, len(cols)),np.tile(cols,len(rows))], matrix.shape)
    return U[idx].sum(), cols

def downArrow(D, matrix):
    rows = set(range(matrix.shape[0]))
    for col in D:
        aux = set(getColCoverage(col, matrix))
        rows = rows.intersection(aux)
    return list(rows)

def upArrow(C, matrix):
    cols = set(range(matrix.shape[1]))
    for row in C:
        aux = set(getRowCoverage(row, matrix))
        cols = cols.intersection(aux)
    return list(cols)

def getColCoverage(col, matrix):
    return np.where(matrix[:,col] == 1)[0]

def getRowCoverage(row, matrix):
    return np.where(matrix[row,:] == 1)[0]

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
