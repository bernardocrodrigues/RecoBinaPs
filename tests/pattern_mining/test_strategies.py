"""
Tests for the fca module.
"""

import pytest
from unittest.mock import patch
import numpy as np
from pattern_mining.strategies import GreConDStrategy, BinaPsStrategy

# pylint: disable=missing-function-docstring


def test_grecond_strategy():

    A = GreConDStrategy(coverage=1.0, dataset_binarization_threshold=1.0)
    B = GreConDStrategy(coverage=0.5, dataset_binarization_threshold=3.0)

    C = BinaPsStrategy(
        dataset_binarization_threshold=3.0,
        epochs=200,
        hidden_dimension_neurons_number=20,
        weights_binarization_threshold=0.5,
    )

    from dataset.common import resolve_folds
    from dataset.movie_lens import load_ml_100k_folds

    data, k_fold = load_ml_100k_folds()
    folds = resolve_folds(data, k_fold)

    fold_index, (trainset, testset) = folds[0]

    # patterns_A = A.mine_patterns(trainset)
    # patterns_B = B.mine_patterns(trainset)
    patterns_C = C.mine_patterns(trainset)

    # print(len(patterns_A))
    # print(len(patterns_B))
    print(len(patterns_C))
    # print(A.actual_coverage)
    # print(B.actual_coverage)
