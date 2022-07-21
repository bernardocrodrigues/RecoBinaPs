# Copyright 2021 Jonas Fischer <fischer@mpi-inf.mpg.de>
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
# This code was based on binaps/Binaps_code/dataLoader.py

import numpy as np
import math

from torch import tensor
from torch.utils.data import Dataset


def parse_dat_file(dat_file_path: str) -> np.array:
    """ Parses and loads a dataset in the .dat format

    The .dat format is one produced by the binaps/Data/Synthetic_data/generate_toy.R script.

    Args:
        dat_file_path: path to the .dat file relative to the current working directory.

    Returns:
        data: numpy matrix with the dataset.

    """

    ncol = -1
    nrow = 0

    with open(dat_file_path) as datF:
        # read .dat format line by line
        l = datF.readline()
        while l:
            # drop newline
            l = l[:-1]
            if l == "":
                continue
            if l[-1] == " ":
                l = l[:-1]
            # get indices as array
            sl = l.split(" ")
            sl = [int(i) for i in sl]
            maxi = max(sl)
            if (ncol < maxi):
                ncol = maxi
            nrow += 1
            l = datF.readline()

    data = np.zeros((nrow, ncol), dtype=np.single)

    with open(dat_file_path) as datF:
        # read .dat format line by line
        l = datF.readline()
        rIdx = 0
        while l:
            # drop newline
            l = l[:-1]
            if l == "":
                continue
            if l[-1] == " ":
                l = l[:-1]
            # get indices as array
            sl = l.split(" ")
            idxs = np.array(sl, dtype=int)
            idxs -= 1
            # assign active cells
            data[rIdx, idxs] = np.repeat(1, idxs.shape[0])
            rIdx += 1
            l = datF.readline()

    return np.asarray(data)


def devide_data(data, proportion):
    """ Given a numpy dataset split randomly its data elements into two datasets.

    Args:
        data: numpy matrix with the dataset
        proportion: proportion (between 1 and 0) in which to split the dataset. '1' means every element goes into the
            first numpy array; '0' means every element goes into the second. '0.5' is an even split and so on.

    Returns:
        (data1, data2): two resulting numpy matrices.

    """

    slice1 = np.arange(0, math.ceil(proportion * data.shape[0]))
    slice2 = np.arange(math.ceil(proportion * data.shape[0]), data.shape[0])

    return data[slice1, :], data[slice2, :]


class BinaryDataset(Dataset):
    """ Binary dataset ready to be consumed by BinaPs """

    def __init__(self, data, device):
        """ Constructor

            Args:
                data: numpy matrix with the dataset.
                device: torch device where the data should be sent.
        """

        self.sparsity = np.count_nonzero(data)/np.prod(data.shape)
        self.data = tensor(data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.data[index, :]

    def matmul(self, other):
        return self.data.matmul(other)

    def nrow(self):
        return self.data.shape[0]

    def ncol(self):
        return self.data.shape[1]

    def getSparsity(self):
        return self.sparsity
