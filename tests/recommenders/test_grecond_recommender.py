# import pytest

# from unittest.mock import patch

# from recommenders import grecond_recommender


# class TestGreConDBiAKNNRecommender:
#     def test_init_invalid_args(self):
#         with pytest.raises(AssertionError):
#             recommender = grecond_recommender.GreConDBiAKNNRecommender(
#                 grecond_coverage="0.8",
#                 dataset_binarization_threshold=0.5,
#             )

#         with pytest.raises(AssertionError):
#             recommender = grecond_recommender.GreConDBiAKNNRecommender(
#                 grecond_coverage=0.8,
#                 dataset_binarization_threshold="0.5",
#             )

#         with pytest.raises(AssertionError):
#             recommender = grecond_recommender.GreConDBiAKNNRecommender(
#                 grecond_coverage=2.0,
#                 dataset_binarization_threshold=0.5,
#             )

#     def test_init_success(self):
#         recommender = grecond_recommender.GreConDBiAKNNRecommender(
#             grecond_coverage=0.8,
#             dataset_binarization_threshold=0.5,
#         )

#         assert recommender.grecond_coverage == 0.8
#         assert recommender.dataset_binarization_threshold == 0.5

#     @patch("recommenders.grecond_recommender.load_binary_dataset_from_trainset")
#     @patch("recommenders.grecond_recommender.grecond")
#     def test_compute_biclusters_from_trainset(
#         self, grecond_mock, load_binary_dataset_from_trainset_mock
#     ):
#         recommender = grecond_recommender.GreConDBiAKNNRecommender(
#             grecond_coverage=0.8,
#             dataset_binarization_threshold=0.5,
#         )

#         load_binary_dataset_from_trainset_mock.return_value = "aaaa"
#         grecond_mock.return_value = ("bbbb", 111)

#         recommender.trainset = 123
#         recommender.compute_biclusters_from_trainset()

#         load_binary_dataset_from_trainset_mock.assert_called_once_with(123, threshold=0.5)
#         grecond_mock.assert_called_once_with("aaaa", coverage=0.8)

#         assert recommender.biclusters == "bbbb"
#         assert recommender.actual_coverage == 111
