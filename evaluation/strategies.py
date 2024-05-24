""" strategies.py

This module contains the implementation of the strategies used to evaluate the
performance of the recommender system. The strategies are divided into two
categories: test and train strategies.

Strategies (strategy design pattern) provide a unified interface to evaluate the
performance of the recommender system.

Client code should use the strategies instead of the standalone functions from
metrics.py.

Copyright 2024 Bernardo C. Rodrigues
See LICENSE file for license details
"""

import statistics

from abc import ABC, abstractmethod
from typing import List, Annotated
from surprise.accuracy import mae, rmse
from surprise.prediction_algorithms import Prediction, AlgoBase
from pydantic import BaseModel, validate_call, ConfigDict

from annotated_types import Gt

from evaluation.metric import (
    get_micro_averaged_recall,
    get_macro_averaged_recall,
    get_recall_at_k,
    get_micro_averaged_precision,
    get_macro_averaged_precision,
    get_precision_at_k,
    count_impossible_predictions,
)


class TestMeasureStrategy(ABC, BaseModel):
    """
    Abstract class for test measure strategies.

    Test measurements happen after the model has been trained and tested.
    It should be used to evaluate the model's performance on the prediction list.
    Therefore, measurements of this type will ingest a list of predictions and
    return a float value.

    """

    @abstractmethod
    def calculate(self, predictions: List[Prediction]) -> float:
        """
        Calculate the measure based on the predictions list.

        Args:
            predictions (List[Prediction]): List of predictions.

        Returns:
            float: The calculated measure value.
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the measure.
        """

    @abstractmethod
    def is_better_higher(self) -> bool:
        """
        Returns whether a higher value of the measure is better.

        This is useful for the evaluation process in order to compare and rank
        different models based on the measure value.

        Returns:
            bool: True if higher values are better, False otherwise.
        """

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def __call__(self, predictions: List[Prediction]) -> float:
        """
        Calculate the measure based on the predictions list.

        Args:
            predictions (List[Prediction]): List of predictions.

        Returns:
            float: The calculated measure value.
        """
        return self.calculate(predictions)


class TrainMeasureStrategy(ABC, BaseModel):
    """
    Abstract class for train measure strategies.

    Train measurements happen after the model has been trained and are based on
    trained model. It should be used to evaluate the model's characteristics
    and perhaps its performance on the training data. Therefore, measurements
    of this type will ingest the model and return a float value.
    """

    @abstractmethod
    def calculate(self, recommender_system: AlgoBase) -> float:
        """
        Calculate the measure based on the recommender system.

        Args:
            recommender_system (AlgoBase): The trained recommender system.

        Returns:
            float: The calculated measure value.
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the measure.
        """

    @abstractmethod
    def is_better_higher(self) -> bool:
        """
        Returns whether a higher value of the measure is better.

        This is useful for the evaluation process in order to compare and rank
        different models based on the measure value.

        Returns:
            bool: True if higher values are better, False otherwise.
        """

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def __call__(self, recommender_system: AlgoBase) -> float:
        """
        Calculate the measure based on the recommender system.

        Args:
            recommender_system (AlgoBase): The trained recommender system.

        Returns:
            float: The calculated measure value.
        """
        return self.calculate(recommender_system)


class MAEStrategy(TestMeasureStrategy):

    verbose: bool = False

    def get_name(self) -> str:
        return "mae"

    def is_better_higher(self) -> bool:
        return False

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return mae(predictions, verbose=self.verbose)


class RMSEStrategy(TestMeasureStrategy):

    verbose: bool = False

    def get_name(self) -> str:
        return "rmse"

    def is_better_higher(self) -> bool:
        return False

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return rmse(predictions, verbose=self.verbose)


class MicroAveragedRecallStrategy(TestMeasureStrategy):

    threshold: Annotated[float, Gt(0.0)] = 1.0

    def get_name(self) -> str:
        return "micro_averaged_recall"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return get_micro_averaged_recall(predictions, threshold=self.threshold)


class MacroAveragedRecallStrategy(TestMeasureStrategy):

    threshold: Annotated[float, Gt(0.0)] = 1.0

    def get_name(self) -> str:
        return "macro_averaged_recall"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return get_macro_averaged_recall(predictions, threshold=self.threshold)


class RecallAtKStrategy(TestMeasureStrategy):

    k: Annotated[int, Gt(0)] = 10
    threshold: Annotated[float, Gt(0.0)] = 1.0

    def get_name(self) -> str:
        return f"recall_at_{self.k}"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return get_recall_at_k(predictions, k=self.k, threshold=self.threshold)


class MicroAveragedPrecisionStrategy(TestMeasureStrategy):

    threshold: Annotated[float, Gt(0.0)] = 1.0

    def get_name(self) -> str:
        return "micro_averaged_precision"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return get_micro_averaged_precision(predictions, threshold=self.threshold)


class MacroAveragedPrecisionStrategy(TestMeasureStrategy):

    threshold: Annotated[float, Gt(0.0)] = 1.0

    def get_name(self) -> str:
        return "macro_averaged_precision"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return get_macro_averaged_precision(predictions, threshold=self.threshold)


class PrecisionAtKStrategy(TestMeasureStrategy):

    k: Annotated[int, Gt(0)] = 10
    threshold: Annotated[float, Gt(0.0)] = 1.0

    def get_name(self) -> str:
        return f"precision_at_{self.k}"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return get_precision_at_k(predictions, k=self.k, threshold=self.threshold)


class CountImpossiblePredictionsStrategy(TestMeasureStrategy):

    def get_name(self) -> str:
        return "count_impossible_predictions"

    def is_better_higher(self) -> bool:
        return False

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return count_impossible_predictions(predictions)


class CoverageStrategy(TrainMeasureStrategy):

    def get_name(self) -> str:
        return "coverage"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:
        return recommender_system.mining_strategy.actual_coverage


class MeanBiclusterSizeStrategy(TrainMeasureStrategy):

    def get_name(self) -> str:
        return "mean_bicluster_size"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:

        mean_bicluster_size = statistics.mean(
            [
                len(bicluster.extent) * len(bicluster.intent)
                for bicluster in recommender_system.biclusters
            ]
        )
        return mean_bicluster_size


class MeanBiclusterIntentStrategy(TrainMeasureStrategy):

    def get_name(self) -> str:
        return "mean_bicluster_intent"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:

        mean_bicluster_intent = statistics.mean(
            [len(bicluster.intent) for bicluster in recommender_system.biclusters]
        )
        return mean_bicluster_intent


class MeanBiclusterExtentStrategy(TrainMeasureStrategy):

    def get_name(self) -> str:
        return "mean_bicluster_extent"

    def is_better_higher(self) -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:

        mean_bicluster_extent = statistics.mean(
            [len(bicluster.extent) for bicluster in recommender_system.biclusters]
        )
        return mean_bicluster_extent
