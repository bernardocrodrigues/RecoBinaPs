""" strategies.py

This module contains the pattern mining strategies. All strategies must implement the
PatternMiningStrategy interface. Recommender algorithms can use these strategies to mine patterns
from the dataset.

Copyright 2024 Bernardo C. Rodrigues
See COPYING file for license details
"""

from abc import ABC, abstractmethod
from typing import Annotated, List, Optional
from tempfile import TemporaryDirectory
from annotated_types import Gt, Le

from surprise import Trainset
from pydantic import BaseModel, validate_call, ConfigDict, Field

from pattern_mining.binaps.binaps_wrapper import run_binaps, get_patterns_from_weights
from pattern_mining.formal_concept_analysis import create_concept
from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
    save_as_binaps_compatible_input,
)

from .formal_concept_analysis import grecond, Concept


class PatternMiningStrategy(ABC):
    """
    Abstract class for the pattern mining strategies.
    """

    @abstractmethod
    def mine_patterns(self, trainset: Trainset) -> List[Concept]:
        """
        Mines the patterns from the dataset.

        Args:
            trainset (Trainset): A surprise trainset.

        Returns:
            List[Concept]: The mined patterns.
        """


class GreConDStrategy(PatternMiningStrategy, BaseModel):
    """
    GreConD strategy for pattern mining.

    This strategy uses the GreConD algorithm to mine the formal concepts from the dataset. Is will
    binarize the dataset according to the dataset_binarization_threshold and then mine the concepts
    with the specified coverage.

    Attributes:
        coverage (float): The coverage of the mined patterns. Must be a float between 0.0 and 1.0.
        dataset_binarization_threshold (float): The threshold for binarizing the dataset.
        actual_coverage (float): The actual coverage of the mined patterns. THis attribute does not
            need to be set during initialization. It will be set after the mine_patterns method is
            called.
    """

    dataset_binarization_threshold: Annotated[float, Gt(0.0)] = 1.0
    coverage: Annotated[float, Gt(0.0), Le(1.0)] = 1.0
    actual_coverage: Optional[float] = None

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def mine_patterns(self, trainset: Trainset) -> List[Concept]:
        """
        Mines the formal concepts from the binary dataset.

        Args:
            binary_dataset (np.ndarray): The binary dataset.

        Returns:
            List[np.ndarray]: The mined patterns.
        """

        binary_dataset = load_binary_dataset_from_trainset(
            trainset, threshold=self.dataset_binarization_threshold
        )

        formal_context, actual_coverage = grecond(binary_dataset, self.coverage)

        self.actual_coverage = actual_coverage

        return formal_context
    
    def __call__(self, trainset: Trainset) -> List[Concept]:
        return self.mine_patterns(trainset)


class BinaPsStrategy(PatternMiningStrategy, BaseModel):
    """
    BinaPs strategy for pattern mining.

    This strategy uses the BinaPs algorithm to mine the formal concepts from the dataset. Is will
    binarize the dataset according to the dataset_binarization_threshold and then mine the concepts.

    """

    dataset_binarization_threshold: Annotated[float, Gt(0.0)] = 1.0
    hidden_dimension_neurons_number: Optional[Annotated[int, Field(gt=0)]] = None
    epochs: Annotated[int, Field(gt=0)] = 100
    weights_binarization_threshold: Annotated[float, Field(ge=0)] = 0.2

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def mine_patterns(self, trainset: Trainset) -> List[Concept]:
        """
        Mines the formal concepts from the binary dataset.

        Args:
            binary_dataset (np.ndarray): The binary dataset.

        Returns:
            List[np.ndarray]: The mined patterns.
        """

        binary_dataset = load_binary_dataset_from_trainset(
            trainset, threshold=self.dataset_binarization_threshold
        )

        with TemporaryDirectory() as temporary_directory:
            with open(f"{temporary_directory}/dataset", "w+", encoding="UTF-8") as file_object:
                save_as_binaps_compatible_input(binary_dataset, file_object)

                weights, _, _ = run_binaps(
                    input_dataset_path=file_object.name,
                    epochs=self.epochs,
                    hidden_dimension=self.hidden_dimension_neurons_number,
                )

        patterns = get_patterns_from_weights(weights, self.weights_binarization_threshold)

        biclusters = []
        for pattern in patterns:
            concept = create_concept([], pattern)
            biclusters.append(concept)

        return biclusters
