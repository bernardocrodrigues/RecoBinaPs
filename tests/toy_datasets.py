"""
toy_datasets.py

This file contains some toy datasets that can be used to test the algorithms. Some of them are taken
from the literature, some of them are created by me. The datasets are:

Bibliography
[1] Mohammed J. Zaki and Wagner Meira, Jr., 'Data Mining and Analysis: Fundamental Concepts and
    Algorithms' <https://dataminingbook.info/first_edition/>
[2] Radim Belohlavek, Vilem Vychodil, 'Discovery of optimal factors in binary data via a novel
    method of matrix decomposition' 
    <https://www.sciencedirect.com/science/article/pii/S0022000009000415>
[3] Elena Nenova, Dmitry I. Ignatov, and Andrey V. Konstantinov, 'An FCA-based Boolean Matrix
    Factorization for Collaborative Filtering 
    <https://publications.hse.ru/pubs/share/folder/2yoq2ezea5/97014436.pdf>
"""


import random
from typing import List
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, Trainset


def convert_raw_rating_list_into_trainset(
    raw_data: List[List[int]], shuffle: bool = True
) -> Trainset:
    """
    Convert a list of raw ratings into a surprise Trainset object.

    Raw data must be a list of lists, where each list is a rating. Each rating must be a list of
    three elements: the first is the user ID, the second is the item ID, the third is the rating.

    Args:
        raw_data (List[List[int]]): The raw data.
        shuffle (bool, optional): If True, the raw data will be shuffled before being converted
            into a Trainset. Defaults to True.

    Returns:
        Trainset: The Trainset object.
    """
    if shuffle:
        random.shuffle(raw_data)
    rating_dataframe = pd.DataFrame(raw_data, columns=["userID", "itemID", "rating"])
    reader = Reader(rating_scale=(0, 5))
    return Dataset.load_from_df(
        rating_dataframe[["userID", "itemID", "rating"]], reader
    ).build_full_trainset()


# fmt: off
my_toy_binary_dataset = np.array(
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
my_toy_dataset_raw_rating = [  # User ID     Item ID     Rating
                                [0,          0,          5],
                                [0,          1,          5],
                                [0,          3,          5],
                                [1,          0,          5],
                                [1,          1,          5],
                                [1,          2,          5],
                                [1,          3,          5],
                                [2,          0,          5],
                                [2,          1,          5],
                                [2,          2,          5],
                                [2,          3,          5],
                                [3,          4,          5],
                                [3,          5,          5],
                                [4,          4,          5],
                                [4,          5,          5],
                                [5,          0,          5],
                                [5,          1,          5],
                                [5,          2,          5],
                                [5,          3,          5],
                                [5,          4,          5],
                                [5,          5,          5],
                                [6,          6,          5],
                            ]
# fmt: on


# fmt: off
my_toy_binary_2_dataset = np.array(
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


# Zaki Dataset
# Dataset taken from Zaki's [1] at page 219
# fmt: off
zaki_binary_dataset = np.array(
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
zaki_dataset_raw_rating = [  # User ID     Item ID     Rating
                              [0,          0,          5],
                              [0,          1,          5],
                              [0,          3,          5],
                              [0,          4,          5],
                              [1,          1,          5],
                              [1,          2,          5],
                              [1,          4,          5],
                              [2,          0,          5],
                              [2,          1,          5],
                              [2,          4,          5],
                              [3,          0,          5],
                              [3,          1,          5],
                              [3,          2,          5],
                              [3,          4,          5],
                              [4,          0,          5],
                              [4,          1,          5],
                              [4,          2,          5],
                              [4,          3,          5],
                              [4,          4,          5],
                              [5,          1,          5],
                              [5,          2,          5],
                              [5,          3,          5],
                        ]
# fmt: on

# Belohlavek Dataset
# Dataset taken from Belohlavek's [2] at page 14
# fmt: off
belohlavek_binary_dataset = np.array(
    [ #     0      1     2      3       4       5
        [True,  False,  True,   False,  True,   True],  # 0
        [False, False,  True,   False,  False,  False], # 1
        [True,  True,   False,  True,   True,   True],  # 2
        [False, False,  True,   False,  False,  True],  # 3
        [False, True,   True,   True,   False,  True],  # 4

    ],
    dtype=bool,
)
belohlavek_dataset_raw_rating = [  # User ID     Item ID     Rating
                                    [0,          0,          5],
                                    [0,          2,          5],
                                    [0,          4,          5],
                                    [0,          5,          5],
                                    [1,          2,          5],
                                    [2,          0,          5],
                                    [2,          1,          5],
                                    [2,          3,          5],
                                    [2,          4,          5],
                                    [2,          5,          5],
                                    [3,          2,          5],
                                    [3,          5,          5],
                                    [4,          1,          5],
                                    [4,          2,          5],
                                    [4,          3,          5],
                                    [4,          5,          5],
                                ]
# fmt: on


# Belohlavek Dataset 2
# Dataset taken from Belohlavek's [2] at page 9
# fmt: off
belohlavek_binary_dataset_2 = np.array(
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


# Nenova Dataset
# Dataset taken from Nenova's [3] at page 62
# fmt: off
nenova_dataset_dataset = np.array(
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
