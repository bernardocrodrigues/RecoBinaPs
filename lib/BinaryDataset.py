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
# Bibliography
# [1] Data Mining and Analysis: Fundamental Concepts and Algorithms <https://dataminingbook.info/first_edition/>
# [2] Discovery of optimal factors in binary data via a novel method of matrix decomposition <https://www.sciencedirect.com/science/article/pii/S0022000009000415>

import numpy as np
from numba import njit


@njit
def _it(binary_dataset: np.array, T: np.array) -> np.array: # pragma: no cover
    """
    Since i and t are algorithmically equivalent only differing if they treat rows as columns (i) or not (t), we provide
    a single implementation that will power both operations.
    """

    if len(T) == 0:
        return np.zeros((0), dtype="int64")

    t = []

    for index, tid in enumerate(binary_dataset):
        result = True
        for x in T:
            result = result and tid[x]
            if not result:
                break
        if result:
            t.append(index)

    return np.array(t)


class BinaryDataset(object):
    """
    This class works as a wrapper over a binary numpy matrix to add some helper functions to aid us perform FCA related
    taks such as computation of intents/extents.
    """
    def __init__(self, binary_dataset: np.array) -> None:
        self._binary_dataset = binary_dataset
        self._transposed_binary_dataset = binary_dataset.T

        self.shape = binary_dataset.shape

        self.X = np.arange(binary_dataset.shape[0], dtype=int)
        self.Y = np.arange(binary_dataset.shape[1], dtype=int)

    def get_raw_copy(self):
        return self._binary_dataset.copy()

    def i(self, T):
        """
        Given [1]'s nomenclature, this method will compute the set of items that are common to all transactions in the
        tidset T.

        Given [2]'s nomenclature, this equivalent to the 'up' operation.
        """
        return _it(self._transposed_binary_dataset, T)

    def t(self, X):
        """Given [1]'s nomenclature, this method will compute the set of all tids that contain all the items in the
        itemset X.

        Given [2]'s nomenclature, this equivalent to the 'down' operation.
        """
        return _it(self._binary_dataset, X)
