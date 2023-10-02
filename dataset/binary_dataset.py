""" binary_dataset.py

This module implements a wrapper over a binary numpy matrix to add some helper functions to aid us
perform FCA related tasks such as computation of intents/extents. It also provides a method to load
a binary dataset from a file that follows the format used by BinAps.

Copyright 2022 Bernardo C. Rodrigues
See LICENSE file for license details

Bibliography
[1] Data Mining and Analysis: Fundamental Concepts and Algorithms 
    <https://dataminingbook.info/first_edition/>
[2] Discovery of optimal factors in binary data via a novel method of matrix decomposition 
    <https://www.sciencedirect.com/science/article/pii/S0022000009000415>

"""

import numpy as np
import numba as nb
from surprise import Trainset
from binaps.original.Binaps_code.dataLoader import readDatFile


@nb.njit
def _it(
    binary_dataset: nb.types.Array, T: nb.types.ListType(nb.types.int64)
) -> nb.types.ListType(nb.types.int64):
    """
    Given a binary dataset and a tidset T, compute the set of items that are common to all
    transactions in the tidset T. This is equivalent to the 'up' operation in [1]. This method is
    implemented using numba to speed up the computation. Since i and t are algorithmically
    equivalent differing only in the way they treat rows as columns and vice-versa, this method is
    used by both i and t. This method is not meant to be called directly, use the i and t methods
    instead.

    Args:
        binary_dataset (np.array): A binary dataset.
        T (np.array): A tidset.

    Returns:
        np.array: The set of items that are common to all transactions in the tidset T.

    """

    if len(T) == 0:
        return nb.typed.List.empty_list(nb.types.int64)

    t = nb.typed.List.empty_list(nb.types.int64) # pylint: disable=invalid-name

    for index, tid in enumerate(binary_dataset):
        result = True
        for x in T:
            result = result and tid[x]
            if not result:
                break
        if result:
            t.append(index)

    return t


class BinaryDataset(object):
    """
    This class works as a wrapper over a binary numpy matrix to add some helper functions to aid us
    perform FCA related tasks such as computation of intents/extents.
    """

    def __init__(self, binary_dataset: np.array) -> None:
        assert isinstance(binary_dataset, np.ndarray)
        assert (
            binary_dataset.dtype == bool
            or binary_dataset.dtype == np.bool_
            or binary_dataset.dtype == nb.types.bool_
        )
        assert binary_dataset.ndim == 2
        assert binary_dataset.size > 0

        self.binary_dataset = binary_dataset
        self.transposed_binary_dataset = binary_dataset.T

        self.shape = binary_dataset.shape

        self.X = np.arange(binary_dataset.shape[0], dtype=int)  # pylint: disable=invalid-name
        self.Y = np.arange(binary_dataset.shape[1], dtype=int)  # pylint: disable=invalid-name

        self.number_of_trues = np.count_nonzero(binary_dataset)
        self.sparsity = self.number_of_trues / binary_dataset.size

    def get_raw_copy(self):
        """
        Get a copy of the binary dataset as a numpy array.

        Returns:
            np.array: A copy of the binary dataset as a numpy array.

        Example:
            dataset = BinaryDataset.load_from_binaps_compatible_input('data.dat')
            raw_dataset = dataset.get_raw_copy()
        """
        return self.binary_dataset.copy()

    def i(self, T):  # pylint: disable=invalid-name
        """
        Given [1]'s nomenclature, this method will compute the set of items that are common to all
        transactions in the tidset T.

        Given [2]'s nomenclature, this equivalent to the 'up' operation.
        """
        return _it(self.transposed_binary_dataset, T)

    def t(self, X):  # pylint: disable=invalid-name
        """Given [1]'s nomenclature, this method will compute the set of all tids that contain all
        the items in the itemset X.

        Given [2]'s nomenclature, this equivalent to the 'down' operation.
        """
        return _it(self.binary_dataset, X)

    @staticmethod
    def load_from_trainset(trainset: Trainset, threshold=1):
        """
        Given a existing trainset, build a binary dataset from it. Note that the item's columns and
        rows will follow the internal trainset representation and will almost certainly differ from
        the original dataset from which the trainset was derived. Use the trainset's to_raw_*
        methods to convert correctly between this spaces.
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

        for row in self.binary_dataset:
            non_zero_indices = np.add(row.nonzero(), 1)[0]
            str_representation = " ".join((str(index) for index in non_zero_indices)) + "\n"
            stream.write(str_representation)
