from pattern_mining.binaps.binaps_wrapper import run_binaps, get_patterns_from_weights
from pydantic import ValidationError
from unittest.mock import patch
import pytest

import torch


class TestRunBinaps:
    def test_invalid_args(self):
        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path=1,
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size="0.8",
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size="32",
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size="32",
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs="15",
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate="0.001",
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay="0.001",
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma="0.9",
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed="42",
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data.csv",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension="1",
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="",
                train_set_size=0.8,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

        with pytest.raises(ValidationError):
            run_binaps(
                input_dataset_path="data",
                train_set_size=2.0,
                batch_size=32,
                test_batch_size=32,
                epochs=15,
                learning_rate=0.001,
                weight_decay=0.001,
                gamma=0.9,
                seed=42,
                hidden_dimension=1,
            )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=-0.1,
                    batch_size=32,
                    test_batch_size=32,
                    epochs=15,
                    learning_rate=0.001,
                    weight_decay=0.001,
                    gamma=0.9,
                    seed=42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=-1,
                    test_batch_size=32,
                    epochs=15,
                    learning_rate=0.001,
                    weight_decay=0.001,
                    gamma=0.9,
                    seed=42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=32,
                    test_batch_size=-1,
                    epochs=15,
                    learning_rate=0.001,
                    weight_decay=0.001,
                    gamma=0.9,
                    seed=42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=32,
                    test_batch_size=32,
                    epochs=-1,
                    learning_rate=0.001,
                    weight_decay=0.001,
                    gamma=0.9,
                    seed=42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=32,
                    test_batch_size=32,
                    epochs=15,
                    learning_rate=-0.001,
                    weight_decay=0.001,
                    gamma=0.9,
                    seed=42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=32,
                    test_batch_size=32,
                    epochs=15,
                    learning_rate=0.001,
                    weight_decay=-0.001,
                    gamma=0.9,
                    seed=42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=32,
                    test_batch_size=32,
                    epochs=15,
                    learning_rate=0.001,
                    weight_decay=0.001,
                    gamma=-0.9,
                    seed=42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=32,
                    test_batch_size=32,
                    epochs=15,
                    learning_rate=0.001,
                    weight_decay=0.001,
                    gamma=0.9,
                    seed=-42,
                    hidden_dimension=1,
                )

            with pytest.raises(ValidationError):
                run_binaps(
                    input_dataset_path="data",
                    train_set_size=0.8,
                    batch_size=32,
                    test_batch_size=32,
                    epochs=15,
                    learning_rate=0.001,
                    weight_decay=0.001,
                    gamma=0.9,
                    seed=42,
                    hidden_dimension=-1,
                )

    @patch("pattern_mining.binaps.binaps_wrapper.learn")
    @patch("pattern_mining.binaps.binaps_wrapper.torch")
    def test_run_binaps_1(self, mock_torch, mock_learn):

        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"
        mock_learn.return_value = ("weights", "train_losses", "test_losses")

        weights, training_losses, test_losses = run_binaps(
            input_dataset_path="data.csv",
            train_set_size=0.8,
            batch_size=32,
            test_batch_size=32,
            epochs=15,
            learning_rate=0.001,
            weight_decay=0.001,
            gamma=0.9,
            seed=42,
            hidden_dimension=1,
        )

        mock_learn.assert_called_with(
            "data.csv", 0.001, 0.9, 0.001, 15, 1, 0.8, 32, 32, "cuda", "cuda"
        )

        mock_torch.cuda.is_available.assert_called()
        mock_torch.device.assert_called_with("cuda")

        assert weights == "weights"
        assert training_losses == "train_losses"
        assert test_losses == "test_losses"

    @patch("pattern_mining.binaps.binaps_wrapper.learn")
    @patch("pattern_mining.binaps.binaps_wrapper.torch")
    def test_run_binaps_2(self, mock_torch, mock_learn):

        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_learn.return_value = ("weights", "train_losses", "test_losses")

        weights, training_losses, test_losses = run_binaps(
            input_dataset_path="dataaaa.csv",
            train_set_size=0.1,
            batch_size=64,
            test_batch_size=11,
            epochs=200,
            learning_rate=0.24,
            weight_decay=0.441,
            gamma=0.12,
            seed=1020,
            hidden_dimension=198,
        )

        mock_learn.assert_called_with(
            "dataaaa.csv", 0.24, 0.12, 0.441, 200, 198, 0.1, 64, 11, "cpu", "cpu"
        )

        mock_torch.cuda.is_available.assert_called()
        mock_torch.device.assert_called_with("cpu")

        assert weights == "weights"
        assert training_losses == "train_losses"
        assert test_losses == "test_losses"


class TestGetPatternsFromWeights:

    def test_invalid_args(self):

        with pytest.raises(ValidationError):
            get_patterns_from_weights("tensor", 0.5)

        with pytest.raises(ValidationError):
            get_patterns_from_weights(torch.tensor([1, 2, 3]), "0.5")

        with pytest.raises(ValidationError):
            get_patterns_from_weights(torch.tensor([1, 2, 3]), -1.0)

    @patch("pattern_mining.binaps.binaps_wrapper.torch")
    @patch("pattern_mining.binaps.binaps_wrapper.BinarizeTensorThresh")
    def test_1(self, mock_binarize_tensor_thresh, mock_torch):

        mock_torch.squeeze.side_effect = [
            torch.tensor([0, 1, 0, 1, 1]),
            torch.tensor([2, 3, 4]),

        ]

        mock_binarize_tensor_thresh.return_value = torch.tensor([0, 1])

        weights = torch.tensor([0.4, 0.5, 0.5, 0.4, 0.5])
        threshold = 0.3

        patterns = get_patterns_from_weights(weights, threshold)

        mock_torch.no_grad.assert_called()
        mock_binarize_tensor_thresh.assert_called_with(weights, threshold)
