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

    dataset = binary_dataset.get_raw_copy()
    dataset = dataset.astype('int')

    # print(dataset)

    aux = GreConD_bin(dataset)
    F = aux.get_concepts()
    current_coverage = coverage

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


import numpy as np
import pandas as pd
import copy


class GreConD_bin:
    def __init__(self, I):   #I--boolean matrix n*m: list of lists of int
        self.I = I
        self.m = len(I[0])
        self.n = len(I)
        self.I_rows = []    # list of integers from binary rows
        self.I_cols = []    # list of integers from binary columns
        for i in self.I:
            self.I_rows.append(int(''.join(map(str,i)), 2))
        for j in range(self.m):
            self.I_cols.append(int(''.join(map(str,[row[j] for row in I])), 2))
        #self.U = np.argwhere(I==1).tolist()
        self.U = copy.deepcopy(self.I_rows)
        self.all_column_indices = list(range(self.m))
        self.concepts = None
        self.factors = None
        self.M_int=None
        self.ones = (1 << self.m) - 1

    def derive_up(self, N):   #N -- list of indices from the set {0,...,n-1}
        if N!=[]:
            intersect_rows = self.I_rows[N[0]]   
            for x in N[1:]:
                intersect_rows &= self.I_rows[x]
            intersect_rows_bin = bin(intersect_rows)[2:]
            self.M_int = intersect_rows 
            len_int_rows = len(intersect_rows_bin)
            M = [self.m-len_int_rows+i for i in range(len_int_rows) if intersect_rows_bin[i]=='1']
            return M
        else:
            return [i for i in range(self.m)]   

    def derive_down(self, M):   #M -- list of indices from the set {0,...,m-1}
        if M!=[]:
            intersect_columns = self.I_cols[M[0]]
            for y in M[1:]:
                intersect_columns &= self.I_cols[y]
            intersect_cols_bin = bin(intersect_columns)[2:]
            len_int_cols = len(intersect_cols_bin)
            N = [self.n-len_int_cols+i for i in range(len_int_cols) if intersect_cols_bin[i]=='1']
            return N
        else:
            return [j for j in range(self.n)]

    def add_elem(self, D, y): #direct sum operation from the algorithm
        D_y = D.copy()
        D_y.append(y)
        C = self.derive_down(D_y)
        D_y = self.derive_up(C)
        D_int = self.M_int
        if C==[] or D_y==[]:
            return [[[],[]], 0, []]
        else:
            len_intersection_U = 0
            for i in C:
                intersect=self.U[i]&D_int
                len_intersection_U += bin(intersect).count('1')
            return [[C, D_y], len_intersection_U, D_int]

    def get_concepts(self):    #returns a set of factor concepts of I
        F = []
        
        while all([v == 0 for v in self.U])!=True:
            D = []
            max_len = 0
            max_len_prev = -1
            max_concept = []
            column_indices = self.all_column_indices
            while column_indices != [] and max_len_prev<max_len:
                max_len_prev = max_len
                for y in column_indices:
                    D_y = self.add_elem(D, y)
                    len_intersection_U = D_y[1]
                    if len_intersection_U>max_len:
                        max_len = len_intersection_U
                        max_concept = D_y[0]
                        D_int = D_y[2]
                D = max_concept[1]
                column_indices = [i for i in column_indices if i not in D]
            F.append(max_concept)
            for i in max_concept[0]:
                self.U[i] = self.U[i] & (~D_int&self.ones)
        self.concepts = F
        return F

    def get_factors(self):    #returns decomposition matrices A and B
        if self.concepts==None:
            concepts = self.get_concepts()
        else:
            concepts = self.concepts
        len_c = len(concepts)
        A = [[0]*len_c for i in range(self.n)]
        B = [[0]*self.m for i in range(len_c)]
        for k in range(len_c):
            for i in concepts[k][0]:
                A[i][k]=1
            for j in concepts[k][1]:
                B[k][j]=1
        self.factors = [A, B]
        return [A, B]

    def multiply_factors(self):    #returns a product of decomposition matrices A and B
        if self.factors==None:
            A_B = self.get_factors()
        else:
            A_B = self.factors
        A = np.array(A_B[0])
        B= np.array(A_B[1])
        I = np.where(np.matmul(A,B)>=1, 1, 0)
        return I