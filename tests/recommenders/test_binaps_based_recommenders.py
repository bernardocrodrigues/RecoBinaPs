# import pytest

# from unittest.mock import patch

# from recommenders import binaps_based_recommenders


# class TestBinaPsKNNRecommender:
#     def test_init_invalid_args(self):
#         with pytest.raises(AssertionError):
#             recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#                 epochs="100",
#                 hidden_dimension_neurons_number=10,
#                 weights_binarization_threshold=0.2,
#                 dataset_binarization_threshold=1.0,
#             )

#         with pytest.raises(AssertionError):
#             recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#                 epochs=100,
#                 hidden_dimension_neurons_number="100",
#                 weights_binarization_threshold=0.2,
#                 dataset_binarization_threshold=1.0,
#             )

#         with pytest.raises(AssertionError):
#             recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#                 epochs=100,
#                 hidden_dimension_neurons_number=-1,
#                 weights_binarization_threshold=0.2,
#                 dataset_binarization_threshold=1.0,
#             )

#         with pytest.raises(AssertionError):
#             recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#                 epochs=100,
#                 hidden_dimension_neurons_number=100,
#                 weights_binarization_threshold="0.2",
#                 dataset_binarization_threshold=1.0,
#             )

#         with pytest.raises(AssertionError):
#             recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#                 epochs=100,
#                 hidden_dimension_neurons_number=100,
#                 weights_binarization_threshold=0.2,
#                 dataset_binarization_threshold="1.0",
#             )

#     def test_init_success(self):
#         recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#             epochs=100,
#             hidden_dimension_neurons_number=10,
#             weights_binarization_threshold=0.2,
#             dataset_binarization_threshold=1.0,
#         )

#         assert recommender.epochs == 100
#         assert recommender.hidden_dimension_neurons_number == 10
#         assert recommender.weights_binarization_threshold == 0.2
#         assert recommender.dataset_binarization_threshold == 1.0

#         recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#             epochs=100,
#             weights_binarization_threshold=0.2,
#             dataset_binarization_threshold=1.0,
#         )

#         assert recommender.epochs == 100
#         assert recommender.hidden_dimension_neurons_number is None
#         assert recommender.weights_binarization_threshold == 0.2
#         assert recommender.dataset_binarization_threshold == 1.0

#     @patch("recommenders.binaps_based_recommenders.get_patterns_from_weights")
#     @patch("recommenders.binaps_based_recommenders.run_binaps")
#     @patch("recommenders.binaps_based_recommenders.save_as_binaps_compatible_input")
#     @patch("recommenders.binaps_based_recommenders.open")
#     @patch("recommenders.binaps_based_recommenders.TemporaryDirectory")
#     @patch("recommenders.binaps_based_recommenders.load_binary_dataset_from_trainset")
#     def test_compute_biclusters_from_trainset_1(
#         self,
#         load_binary_dataset_from_trainset_mock,
#         temporary_directory_mock,
#         open_mock,
#         save_as_binaps_compatible_input_mock,
#         run_binaps_mock,
#         get_patterns_from_weights_mock,
#     ):
#         recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#             epochs=100,
#             weights_binarization_threshold=0.2,
#             dataset_binarization_threshold=0.7,
#         )
#         recommender.trainset = 'trainset'

#         load_binary_dataset_from_trainset_mock.return_value = "binary_dataset"
#         temporary_directory_mock.return_value.__enter__.return_value = "temporary_directory"
#         run_binaps_mock.return_value = ("weights", "training_loss", "test_loss")
#         get_patterns_from_weights_mock.return_value = "biclusters"

#         recommender.compute_biclusters_from_trainset()

#         load_binary_dataset_from_trainset_mock.assert_called_once_with(
#             'trainset', threshold=0.7
#         )

#         temporary_directory_mock.assert_called_once_with()

#         open_mock.assert_called_once_with(
#             "temporary_directory/dataset", "w+", encoding="UTF-8"
#         )

#         save_as_binaps_compatible_input_mock.assert_called_once_with(
#             "binary_dataset", open_mock.return_value.__enter__.return_value
#         )

#         run_binaps_mock.assert_called_once_with(
#             input_dataset_path=open_mock.return_value.__enter__.return_value.name,
#             epochs=100,
#             hidden_dimension=-1,
#         )

#         get_patterns_from_weights_mock.assert_called_once_with("weights", 0.2)

#         assert recommender.biclusters == "biclusters"

#     @patch("recommenders.binaps_based_recommenders.get_patterns_from_weights")
#     @patch("recommenders.binaps_based_recommenders.run_binaps")
#     @patch("recommenders.binaps_based_recommenders.save_as_binaps_compatible_input")
#     @patch("recommenders.binaps_based_recommenders.open")
#     @patch("recommenders.binaps_based_recommenders.TemporaryDirectory")
#     @patch("recommenders.binaps_based_recommenders.load_binary_dataset_from_trainset")
#     def test_compute_biclusters_from_trainset_2(
#         self,
#         load_binary_dataset_from_trainset_mock,
#         temporary_directory_mock,
#         open_mock,
#         save_as_binaps_compatible_input_mock,
#         run_binaps_mock,
#         get_patterns_from_weights_mock,
#     ):
#         recommender = binaps_based_recommenders.BinaPsKNNRecommender(
#             epochs=100,
#             hidden_dimension_neurons_number=10,
#             weights_binarization_threshold=0.2,
#             dataset_binarization_threshold=0.7,
#         )
#         recommender.trainset = 'trainset'

#         load_binary_dataset_from_trainset_mock.return_value = "binary_dataset"
#         temporary_directory_mock.return_value.__enter__.return_value = "temporary_directory"
#         run_binaps_mock.return_value = ("weights", "training_loss", "test_loss")
#         get_patterns_from_weights_mock.return_value = "biclusters"

#         recommender.compute_biclusters_from_trainset()

#         load_binary_dataset_from_trainset_mock.assert_called_once_with(
#             'trainset', threshold=0.7
#         )

#         temporary_directory_mock.assert_called_once_with()

#         open_mock.assert_called_once_with(
#             "temporary_directory/dataset", "w+", encoding="UTF-8"
#         )

#         save_as_binaps_compatible_input_mock.assert_called_once_with(
#             "binary_dataset", open_mock.return_value.__enter__.return_value
#         )

#         run_binaps_mock.assert_called_once_with(
#             input_dataset_path=open_mock.return_value.__enter__.return_value.name,
#             epochs=100,
#             hidden_dimension=10,
#         )

#         get_patterns_from_weights_mock.assert_called_once_with("weights", 0.2)

#         assert recommender.biclusters == "biclusters"
