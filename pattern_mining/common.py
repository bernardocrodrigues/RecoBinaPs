""" common.py

Common functions used by the pattern mining module.

"""
from typing import List
import numpy as np


def filter_patterns_based_on_bicluster_sparsity(
    ratings_dataset: np.ndarray, patterns: List[np.ndarray], threshold: float = 0.5
) -> List[np.ndarray]:
    """
    Filters the patterns based on the sparsity of the biclusters they form.

    Args:
        ratings_dataset (np.ndarray): The ratings dataset.
        patterns (List[np.ndarray]): The patterns to filter.
        threshold (float): The threshold to use for filtering.

    Returns:
        List[np.ndarray]: The filtered patterns.
    """

    assert isinstance(ratings_dataset, np.ndarray)
    assert ratings_dataset.ndim == 2
    assert ratings_dataset.shape[0] > 0
    assert ratings_dataset.shape[1] > 0
    assert np.issubdtype(ratings_dataset.dtype, np.number)
    assert isinstance(patterns, list)
    assert all(isinstance(pattern, np.ndarray) for pattern in patterns)
    assert all(pattern.ndim == 1 for pattern in patterns)

    assert all(all(item <= ratings_dataset.shape[1] for item in pattern) for pattern in patterns)

    assert isinstance(threshold, float)
    assert 0 <= threshold <= 1

    filtered_patterns = []

    for pattern in patterns:
        bicluster = ratings_dataset[:, pattern]
        bicluster_sparsity = (bicluster > 0).sum() / bicluster.size

        if bicluster_sparsity >= threshold:
            filtered_patterns.append(pattern)

    return filtered_patterns
