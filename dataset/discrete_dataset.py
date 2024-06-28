""" discrete_dataset.py

This module implements a wrapper over a discrete numpy matrix.

Copyright 2024 Bernardo C. Rodrigues
See LICENSE file for license details


"""

import numpy as np
import numba as nb
import typing
from surprise import Trainset
from pattern_mining.binaps.original.Binaps_code.dataLoader import readDatFile



def load_discrete_dataset_from_trainset(trainset: Trainset):
    """
    Given a existing trainset, build a discrete dataset from it. Note that the item's columns and
    rows will follow the internal trainset representation and will almost certainly differ from
    the original dataset from which the trainset was derived. Use the trainset's to_raw_*
    methods to convert correctly between this spaces.

    Args:
        trainset (Trainset): A surprise Trainset object.

    Returns:
        np.array: A discrete dataset
    """

    assert isinstance(trainset, Trainset)

    dataset = np.zeros((trainset.n_users, trainset.n_items), dtype=int)

    for uid, iid, rating in trainset.all_ratings():
        dataset[uid][iid] = round(rating)

    return dataset


def save_as_qubic2_compatible_input(dataset: np.ndarray, stream: typing.IO):
    """
    Save the discrete dataset as a qubic2-compatible input.

    Args:
        stream: A file-like object to write the qubic2-compatible input.

    """

    header = "p\t" + "\t".join(str(i) for i in range(dataset.shape[1]))
    row_names = np.char.array([str(i) for i in range(dataset.shape[0])])
    dataset = dataset.astype(str)
    dataset = np.hstack((row_names[:, np.newaxis], dataset))

    np.savetxt(stream, dataset, delimiter="\t", header=header, fmt="%s", comments="")
