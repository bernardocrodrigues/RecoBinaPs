from pattern_mining.common import apply_bicluster_sparsity_filter
import numpy as np


class Test_filter_patterns_based_on_bicluster_sparsity:
    def test_filter_patterns_based_on_bicluster_sparsity(self):
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

        patterns = [
            np.array([0, 1]),
            np.array([0]),
            np.array([4]),
        ]

        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.2
        )
        assert len(filtered_patterns) == 3

        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.1
        )
        assert len(filtered_patterns) == 3

        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.3
        )
        assert len(filtered_patterns) == 0

        patterns = [
            np.array([2]),
        ]

        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.2
        )
        assert len(filtered_patterns) == 1
        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.6
        )
        assert len(filtered_patterns) == 1
        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.7
        )
        assert len(filtered_patterns) == 0

        patterns = [
            np.array([0, 1]),
            np.array([0]),
            np.array([4]),
            np.array([2]),
        ]

        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.2
        )
        assert len(filtered_patterns) == 4
        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.6
        )
        assert len(filtered_patterns) == 1
        filtered_patterns = apply_bicluster_sparsity_filter(
            rating_dataset, patterns, 0.7
        )
        assert len(filtered_patterns) == 0