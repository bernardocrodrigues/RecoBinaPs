import pytest

from pattern_mining.common import (
    _validade_args,
    apply_bicluster_sparsity_filter,
    apply_bicluster_coverage_filter,
)
from pattern_mining.formal_concept_analysis import create_concept
import numpy as np


class TestValidadeArgs:
    ratings_dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    biclusters = [
        create_concept(np.array([0, 1, 2], dtype=np.int64), np.array([0, 1], dtype=np.int64)),
        create_concept(np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64)),
        create_concept(np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)),
        create_concept(np.array([0, 1], dtype=np.int64), np.array([2], dtype=np.int64)),
    ]

    def test_success(self):
        _validade_args(self.ratings_dataset, self.biclusters, 0.5)

    def test_failure_1(self):
        with pytest.raises(AssertionError):
            _validade_args("not an array", self.biclusters, 0.5)

    def test_failure_2(self):
        with pytest.raises(AssertionError):
            _validade_args(np.array([]), self.biclusters, 0.5)

    def test_failure_3(self):
        with pytest.raises(AssertionError):
            _validade_args(np.array([[]]), self.biclusters, 0.5)

    def test_failure_4(self):
        with pytest.raises(AssertionError):
            _validade_args(self.ratings_dataset.astype(np.int64), self.biclusters, 0.5)

    def test_failure_5(self):
        with pytest.raises(AssertionError):
            _validade_args(self.ratings_dataset, "not a list", 0.5)

    def test_failure_6(self):
        with pytest.raises(AssertionError):
            _validade_args(self.ratings_dataset, [1, 2, 3], 0.5)

    def test_failure_7(self):
        with pytest.raises(AssertionError):
            _validade_args(
                self.ratings_dataset,
                [
                    create_concept(
                        np.array([0, 1, 20], dtype=np.int64), np.array([0, 10], dtype=np.int64)
                    )
                ],
                0.5,
            )

    def test_failure_8(self):
        with pytest.raises(AssertionError):
            _validade_args(self.ratings_dataset, self.biclusters, "0.5")

    def test_failure_9(self):
        with pytest.raises(AssertionError):
            _validade_args(self.ratings_dataset, self.biclusters, -0.5)

        with pytest.raises(AssertionError):
            _validade_args(self.ratings_dataset, self.biclusters, 1.5)


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


class TestApplyBiclusterCoverageFilter:

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
            create_concept(
                np.array([0, 1, 2, 3, 4], dtype=np.int64), np.array([2], dtype=np.int64)
            ),
            create_concept(
                np.array([2], dtype=np.int64), np.array([0, 1, 2, 3, 4], dtype=np.int64)
            ),
            create_concept(
                np.array([0, 2], dtype=np.int64), np.array([0, 1, 2], dtype=np.int64)
            ),
        ]

        filtered_patterns = apply_bicluster_coverage_filter(self.rating_dataset, patterns, 0.6)
        assert len(filtered_patterns) == 0

        filtered_patterns = apply_bicluster_coverage_filter(self.rating_dataset, patterns, 0.5)
        assert len(filtered_patterns) == 2

        filtered_patterns = apply_bicluster_coverage_filter(self.rating_dataset, patterns, 0.2)
        assert len(filtered_patterns) == 4

        filtered_patterns = apply_bicluster_coverage_filter(self.rating_dataset, patterns, 0.1)
        assert len(filtered_patterns) == 6

