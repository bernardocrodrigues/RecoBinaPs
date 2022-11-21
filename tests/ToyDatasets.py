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
# This module comprises a series o example datasets to be used in tests and demos. Some of them are taken from books or
# papers which enable us to cross check if our implementation is doing what it is supposed to given what the literature
# reports.
#
# Bibliography
# [1] Mohammed J. Zaki and Wagner Meira, Jr., 'Data Mining and Analysis: Fundamental Concepts and Algorithms' 
#     <https://dataminingbook.info/first_edition/>
# [2] Radim Belohlavek, Vilem Vychodil, 'Discovery of optimal factors in binary data via a novel method of matrix
#     decomposition' <https://www.sciencedirect.com/science/article/pii/S0022000009000415>
# [3] Elena Nenova, Dmitry I. Ignatov, and Andrey V. Konstantinov, 'An FCA-based Boolean Matrix Factorisation for
#     Collaborative Filtering <https://publications.hse.ru/pubs/share/folder/2yoq2ezea5/97014436.pdf>

import numpy as np
from lib.BinaryDataset import BinaryDataset

# fmt: off
my_toy_dataset_raw = np.array(
    [#   0      1       2       3       4       5       6
        [True,  True,   False,  True,   False,  False,  False], # 0
        [True,  True,   True,   True,   False,  False,  False], # 1
        [True,  True,   True,   True,   False,  False,  False], # 2
        [False, False,  False,  False,  True,   True,   False], # 3
        [False, False,  False,  False,  True,   True,   False], # 4
        [True,  True,   True,   True,   True,   True,   False], # 5
        [False, False,  False,  False,  False,  False,  True ], # 6
    ],
    dtype=bool,
)
# fmt: on
my_toy_binary_dataset = BinaryDataset(my_toy_dataset_raw)

# fmt: off
my_toy_dataset_2_raw = np.array(
    [#   0      1       2       3       4       5       6
        [True,  True,   False,  True,   False,  False,  False], # 0
        [True,  True,   True,   True,   False,  False,  False], # 1
        [True,  True,   True,   True,   False,  False,  False], # 2
        [False, False,  False,  False,  True,   False,  False], # 3
        [False, False,  False,  False,  True,   False,  False], # 4
        [True,  True,   True,   True,   True,   False,  False], # 5
        [False, False,  False,  False,  False,  False,  True ], # 6
        [False, False,  False,  False,  False,  False,  False ],# 7
    ],
    dtype=bool,
)
# fmt: on
my_toy_binary_2_dataset = BinaryDataset(my_toy_dataset_2_raw)

# Zaki Dataset
# Dataset taken from Zaki's [1] at page 219
# fmt: off
zaki_dataset_raw = np.array(
    [ #  0      1     2      3      4
        [True,  True, False, True,  True ],  # 0
        [False, True, True,  False, True ],  # 1
        [True,  True, False, False, True ],  # 2
        [True,  True, True,  False, True ],  # 3
        [True,  True, True,  True,  True ],  # 4
        [False, True, True,  True,  False],  # 5
    ],
    dtype=bool,
)
# fmt: on
zaki_binary_dataset = BinaryDataset(zaki_dataset_raw)

# Belohlavek Dataset
# Dataset taken from Belohlavek's [2] at page 14
# fmt: off
belohlavek_dataset_raw = np.array(
    [ #     0      1     2      3       4       5
        [True,  False,  True,   False,  True,   True],  # 0
        [False, False,  True,   False,  False,  False], # 1
        [True,  True,   False,  True,   True,   True],  # 2
        [False, False,  True,   False,  False,  True],  # 3
        [False, True,   True,   True,   False,  True],  # 4

    ],
    dtype=bool,
)
# fmt: on
belohlavek_binary_dataset = BinaryDataset(belohlavek_dataset_raw)

# Belohlavek Dataset 2
# Dataset taken from Belohlavek's [2] at page 9
# fmt: off
belohlavek_dataset_raw_2 = np.array(
    [ #     0       1       2       3       4       5       6       7
        [True,  True,   True,   False,  True,   False,  False,  False],  # 0
        [True,  True,   False,  False,  False,  True,   False,  True],   # 1
        [False, True,   False,  False,  True,   False,  True,   False],  # 2
        [True,  True,   False,  False,  False,  True,   False,  True],   # 3
        [True,  True,   True,   False,  True,   False,  False,  False],  # 4
        [False, True,   False,  False,  True,   False,  True,   False],  # 5
        [False, True,   False,  False,  True,   False,  True,   False],  # 6
        [False, False,  False,  False,  False,  False,  True,   False],  # 7
        [True,  True,   True,   False,  True,   False,  False,  False],  # 8
        [False, False,  False,  False,  False,  False,  True,   False],  # 9
        [True,  True,   True,   False,  True,   False,  False,  False],  # 10
        [True,  True,   False,  False,  False,  True,   False,  True],  # 11


    ],
    dtype=bool,
)
# fmt: on
belohlavek_binary_dataset_2 = BinaryDataset(belohlavek_dataset_raw_2)

# Nenova Dataset
# Dataset taken from Nenova's [3] at page 62
# fmt: off
nenova_dataset_raw = np.array(
    [ #  0      1       2       3       4       5       6
        [True,  True,   True,   False,  False,  False,  False], # 0
        [True,  True,   True,   True,   True,   False,  False], # 1
        [False, False,  False,  True,   True,   False,  False], # 2
        [False, False,  False,  True,   True,   True,   True],  # 3
        [False, False,  False,  False,  False,  True,   True],  # 4
        [False, False,  False,  False,  False,  True,   True],  # 5


    ],
    dtype=bool,
)
# fmt: on
nenova_dataset_dataset = BinaryDataset(nenova_dataset_raw)
