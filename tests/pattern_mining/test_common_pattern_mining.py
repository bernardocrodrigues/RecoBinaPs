import pytest

from pattern_mining.common import (
    _validade_args,
    apply_bicluster_sparsity_filter,
    apply_bicluster_coverage_filter,
)
from pattern_mining.formal_concept_analysis import create_concept
import numpy as np

class TestApplyBiclusterSparsityFilter:
    rating_dataset = np.array(
        [
            [1.2, 2.3, 0.0, 0.0, 0.0],  # User 1
            [0.0, 0.0, 2.3, 0.0, 0.0],  # User 2
            [0.0, 0.0, 1.1, 0.0, 5.0],  # User 3
            [0.0, 0.0, 0.0, 0.0, 0.0],  # User 4
            [0.0, 0.0, 1.0, 0.0, 0.0],  # User 5
        ],
        dtype=np.float64,
    )

    def test_1(self):
        patterns = [
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([0, 1], dtype=np.int64)
            ),
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([0], dtype=np.int64)
            ),
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([4], dtype=np.int64)
            ),
        ]

        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.2)
        assert len(filtered_patterns) == 3

        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.1)
        assert len(filtered_patterns) == 3

        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.3)
        assert len(filtered_patterns) == 0

    def test_2(self):
        patterns = [
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([2], dtype=np.int64)
            ),
        ]

        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.2)
        assert len(filtered_patterns) == 1
        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.6)
        assert len(filtered_patterns) == 1
        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.7)
        assert len(filtered_patterns) == 0

    def test_3(self):
        patterns = [
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([0, 1], dtype=np.int64)
            ),
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([0], dtype=np.int64)
            ),
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([4], dtype=np.int64)
            ),
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([2], dtype=np.int64)
            ),
        ]

        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.2)
        assert len(filtered_patterns) == 4
        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.6)
        assert len(filtered_patterns) == 1
        filtered_patterns = apply_bicluster_sparsity_filter(self.rating_dataset, patterns, 0.7)
        assert len(filtered_patterns) == 0
