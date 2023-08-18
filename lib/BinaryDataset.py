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
from surprise import Trainset
from binaps.Binaps_code.dataLoader import readDatFile


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

        self.number_of_trues = np.count_nonzero(binary_dataset)
        self.sparsity = self.number_of_trues / binary_dataset.size

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

    @staticmethod
    def load_from_trainset(trainset: Trainset, threshold=1):
        """
        Given a existing trainset, build a binary dataset from it. Note that the item's columns and rows will follow the
        internal trainset representation and will almost certainly differ from the original dataset from which the
        trainset was derived. Use the trainset's to_raw_* methods to convert correctly between this spaces.
        """

        raw_dataset = np.zeros((trainset.n_users, trainset.n_items), bool)

        for uid, iid, rating in trainset.all_ratings():
            if rating >= threshold:
                raw_dataset[uid][iid] = True

        return BinaryDataset(raw_dataset)
    
    @staticmethod
    def load_from_binaps_compatible_input(file_path):
        """
        Instantiate a binary dataset based on the contents of a file that follows the format 
        used by BinAps.

        Args:
            file_path (str): The path to the file containing the binary dataset.

        Returns:
            BinaryDataset: A BinaryDataset object initialized with the data from the file.

        Raises:
            FileNotFoundError: If the specified file_path does not exist.
            IOError: If there is an error reading the file.

        Example:
            dataset = BinaryDataset.load_from_binaps_compatible_input('data.dat')
        """
        raw_dataset = readDatFile(file_path)
        return BinaryDataset(raw_dataset.astype(bool))


    def save_as_binaps_compatible_input(self, stream):   
        """
        Save the binary dataset as a binaps-compatible input.

        Args:
            stream: A file-like object to write the binaps-compatible input.

        Returns:
            None

        This method takes the binary dataset and writes its binaps-compatible representation
        to the provided stream. Each row of the binary dataset is processed, and the indices
        of non-zero elements in the row are converted to a string representation. The string
        representations of all rows are concatenated with a newline character and written to
        the stream.

        Example:
            binary_dataset = np.array([[1, 0, 1],
                                       [0, 1, 0],
                                       [1, 1, 0]])

            with open('binaps_input.txt', 'w') as stream:
                save_as_binaps_compatible_input(binary_dataset, stream)

            # Contents of 'binaps_input.txt':
            # 1 3
            # 2
            # 1 2
        """

        for row in self._binary_dataset:
            non_zero_indices = np.add(row.nonzero(), 1)[0]
            str_representation =  ' '.join((str(indice) for indice in non_zero_indices)) + '\n'
            stream.write(str_representation)
