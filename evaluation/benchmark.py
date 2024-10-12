import time

from abc import ABC, abstractmethod
from json import JSONEncoder
from typing import List, Tuple, Type
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numba as nb
from surprise import AlgoBase, Trainset
from pydantic import validate_call, ConfigDict

from evaluation.strategies import TestMeasureStrategy, TrainMeasureStrategy


def print_progress(tasks: List) -> None:
    """
    Print progress of a list of tasks.
    """

    total = len(tasks)
    start_time = time.time()

    # enumeration starts at 1 to avoid division by zero
    for i, _ in enumerate(as_completed(tasks), 1):

        now = time.time()
        elapsed_time = now - start_time

        average_time = elapsed_time / i
        estimated_time_left = average_time * (total - i)

        average_minutes = average_time // 60
        average_seconds = average_time % 60

        time_left_minutes = estimated_time_left // 60
        time_left_seconds = estimated_time_left % 60

        estimated_completion_time = now + estimated_time_left
        estimated_completion_time -= 3 * 3600  # account for timezone

        print(
            f"Completed {i}/{total} | "
            f"Avg. time/task: {int(average_minutes)}m {average_seconds:.1f}s | "
            f"Time left: {int(time_left_minutes)}m {time_left_seconds:.1f}s | "
            f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(estimated_completion_time))}     ",
            flush=True,
            end="\r",
        )

    print("\nAll tasks completed.")
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"Total time: {hours}h {minutes}m {seconds:.1f}s", flush=True, end="\r")


def iterator_progress(tasks):
    """
    A iterator wrapper that prints progress of a list of tasks.
    """

    total = len(tasks)
    start_time = time.time()

    def update(task_number):

        now = time.time()
        elapsed_time = now - start_time

        average_time = elapsed_time / task_number
        estimated_time_left = average_time * (total - task_number)

        average_minutes = average_time // 60
        average_seconds = average_time % 60

        time_left_minutes = estimated_time_left // 60
        time_left_seconds = estimated_time_left % 60

        print(
            f"Completed {task_number}/{total} | "
            f"Avg. time/task: {int(average_minutes)}m {average_seconds:.1f}s | "
            f"Time left: {int(time_left_minutes)}m {time_left_seconds:.1f}s",
            end="\r",
            flush=True,
        )

    update(0.1)  # avoid div/0

    for i, item in enumerate(tasks):
        yield item
        update(i + 1)

    print("\nAll tasks completed.", flush=True)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print(f"Total time: {hours}h {minutes}m {seconds:.1f}s", flush=True)


def generate_parameter_combinations(parameters_grid: dict) -> List[dict]:
    """
    Generate all possible combinations as a list of named arguments. This is similar
    to itertools.product but returns a list of named arguments instead of tuples.

    Args:
    parameters_grid: dict
        A dictionary of parameter names and their possible values.
        Ex. {"a": [1, 2], "b": [3, 4]}

    Returns:
    List[dict]
        A list of named arguments.
        Ex. [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]

    """
    named_args = []
    keys = list(parameters_grid.keys())
    for combination in product(*parameters_grid.values()):
        named_args.append({keys[i]: combination[i] for i in range(len(keys))})
    return named_args


class FallbackEncoder(JSONEncoder):
    """
    A custom JSON encoder that handles serialization of objects.

    This encoder extends the JSONEncoder class and provides a custom implementation
    of the `default` method to handle serialization of objects that are not natively
    serializable by the JSONEncoder.

    If the object is an instance of `nb.core.registry.CPUDispatcher`, it returns the
    object's name. Otherwise, it serializes the object by creating a dictionary with
    the object's class name and its attributes.

    Args:
        JSONEncoder (class): The base JSONEncoder class.

    Returns:
        str: The serialized JSON string representation of the object.
    """

    def default(self, obj):
        try:
            super().default(obj)
        except TypeError:
            if isinstance(obj, nb.core.registry.CPUDispatcher):
                return obj.__name__
            else:
                serialized_object = {"name": obj.__class__.__name__}
                serialized_object.update(obj.__dict__)
                return serialized_object


def determine_worker_split(tasks: List, max_workers: int):
    """
    Determine the number of parent and child workers for parallelization.

    This function is used to determine how to split the workers when parallelizing
    tasks in a nested fashion. The idea is to have a number of parent workers that
    is less than or equal to the number of tasks, and then split the remaining
    workers among the tasks.

    Args:
    tasks: List
        The tasks to parallelize.
    max_workers: int
        The maximum number of workers to use.

    Returns:
    Tuple[int, List[int]]
        The number of parent workers and a list of child workers.
    """
    if len(tasks) < max_workers:
        parent_workers = len(tasks)
        whole_child_workers = max_workers // len(tasks)
        workers_left = max_workers % len(tasks)

        child_workers = [whole_child_workers] * len(tasks)

        for i in range(workers_left):
            child_workers[i] += 1
    else:
        parent_workers = max_workers
        child_workers = [1] * len(tasks)
    return parent_workers, child_workers


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def fit_and_score(
    recommender_system: AlgoBase,
    trainset: Trainset,
    testset: List[Tuple[str, str, float]],
    test_measures: List[TestMeasureStrategy] = [],
    train_measures: List[TrainMeasureStrategy] = [],
    verbose=True,
) -> Tuple[dict, dict, float, float]:
    """
    Fit a recommender system on a trainset and evaluate it on a testset.

    This function is based on Surprise's fit_and_score
    (surprise.model_selection.validation.fit_and_score) however it is somewhat
    simplified and adapted to:
    - work with our custom measures (modeled as strategies)
    - process precomputed trainset and testset

    Args:
    recommender_system: AlgoBase
        The recommender system to fit and evaluate.
    trainset: Trainset
        The training dataset.
    testset: List[Tuple[str, str, float]]
        The test dataset.
    test_measures: List[TestMeasureStrategy]
        The test measures to compute. A list
    train_measures: List[TrainMeasureStrategy]
        The train measures to compute.

    Returns:
    Tuple[dict, dict, float, float]
        The test measurements, train measurements, fit time and test time.

    """
    start_fit = time.time()
    recommender_system.fit(trainset)
    fit_time = time.time() - start_fit

    train_measurements = {}
    for measure in train_measures:
        measure_name = measure.get_name()
        try:
            train_measurements[measure_name] = measure.calculate(recommender_system)
        except ValueError:
            train_measurements[measure_name] = None

    if verbose:
        testset = iterator_progress(testset)

    start_test = time.time()
    predictions = recommender_system.test(testset)
    test_time = time.time() - start_test

    test_measurements = {}

    for measure in test_measures:
        measure_name = measure.get_name()
        try:
            test_measurements[measure_name] = measure.calculate(predictions)
        except ValueError:
            test_measurements[measure_name] = None

    return test_measurements, train_measurements, fit_time, test_time


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def cross_validate(
    recommender_system: AlgoBase,
    folds: List[Tuple[Trainset, List[Tuple[str, str, float]]]],
    test_measures: List[TestMeasureStrategy] = [],
    train_measures: List[TrainMeasureStrategy] = [],
    max_workers=1,
    verbose=True,
):
    """
    Cross validate a recommender system.

    This function is based on Surprise's cross_validate
    (surprise.model_selection.validation.cross_validate) however it is somewhat
    simplified and adapted to:
    - work with our custom measures (modeled as strategies)
    - process precomputed trainset and testset
    - uses futures instead of parallel for parallelization
    """

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                fit_and_score,
                recommender_system,
                trainset,
                testset,
                test_measures,
                train_measures,
                False,
            )
            for trainset, testset in folds
        ]

        if verbose:
            print_progress(futures)

        out = [future.result() for future in futures]

    (test_measures_dicts, train_measures_dicts, fit_times, test_times) = zip(*out)

    measurements = {}

    for train_measure in train_measures:
        train_measure_name = train_measure.get_name()
        measurements[train_measure_name] = np.asarray(
            [d[train_measure_name] for d in train_measures_dicts]
        )

    for test_measure in test_measures:
        test_measure_name = test_measure.get_name()
        measurements[test_measure_name] = np.asarray(
            [d[test_measure_name] for d in test_measures_dicts]
        )

    measurements["fit_time"] = list(fit_times)
    measurements["test_time"] = list(test_times)

    return measurements


def cross_validade_recommenders(
    recommenders: List[AlgoBase],
    folds: List[Tuple[Trainset, List[Tuple[str, str, float]]]],
    test_measures: List[TestMeasureStrategy] = [],
    train_measures: List[TrainMeasureStrategy] = [],
    max_workers=1,
    verbose=True,
):
    """
    Cross validate a list of recommender systems.

    Args:
    recommenders: List[AlgoBase]
        The recommender systems to cross validate.
    folds: List[Tuple[Trainset, List[Tuple[str, str, float]]]]
        The train-test splits.
    test_measures: List[TestMeasureStrategy]
        The test measures to compute.
    train_measures: List[TrainMeasureStrategy]
        The train measures to compute.
    max_workers: int
        The maximum number of workers to use.
    verbose: bool
        Whether to print progress.

    Returns:
    Dict[AlgoBase, dict]
        A dictionary mapping each recommender to its measurements.
    """

    parent_workers, child_workers = determine_worker_split(recommenders, max_workers)

    with ProcessPoolExecutor(max_workers=parent_workers) as executor:

        futures = [
            executor.submit(
                cross_validate,
                recommender,
                folds,
                test_measures,
                train_measures,
                child_workers[id],
                False,
            )
            for id, recommender in enumerate(recommenders)
        ]

        if verbose:
            print_progress(futures)

        output = [future.result() for future in futures]

    measurements = [recommender_measurements for recommender_measurements in output]

    return measurements


class BaseSearch(ABC):
    """Base class for hyper parameter search with cross-validation."""

    @abstractmethod
    def __init__(
        self,
        recommender_class: Type[AlgoBase],
        param_grid: dict,
        test_measures: List[TestMeasureStrategy] = [],
        train_measures: List[TrainMeasureStrategy] = [],
        max_workers=1,
    ):
        self.recommender_class = recommender_class
        self.param_grid = param_grid
        self.test_measures = test_measures
        self.train_measures = train_measures
        self.max_workers = max_workers
        self.param_combinations = None
        self.recommenders_measurements = None

        self.recommenders_mean_measurements = None

        self.raw = None
        self.ordering = None
        self.best = None

    def fit(self, folds):

        recommenders = [self.recommender_class(**params) for params in self.param_combinations]

        self.recommenders_measurements = cross_validade_recommenders(
            recommenders=recommenders,
            folds=folds,
            test_measures=self.test_measures,
            train_measures=self.train_measures,
            max_workers=self.max_workers,
            verbose=True,
        )

        return self.compute_results()

    def compute_mean_measurements(self):
        """
        Compute the mean measurements for each recommender raw measurements.

        If there are None values in the raw measurements, the mean value is set to None.

        Example:
        raw_measurements = [
            {"fit_time": [1, 2, 3], "test_time": [1, 2, 3], "RMSE": [1, 2, 3]},
            {"fit_time": [3, 4, 5], "test_time": [3, 4, 5], "RMSE": [3, 4, None]},
        ]

        mean_measurements = [
            {"fit_time": 2, "test_time": 2, "RMSE": 2},
            {"fit_time": 4, "test_time": 4, "RMSE": None},
        ]

        Returns:
            list: A list of dictionaries containing the mean measurements for each recommender.
        """

        recommenders_mean_measurements = []

        for recommender_measurements in self.recommenders_measurements:
            recommender_mean_measurements = {}
            for measure, measurements in recommender_measurements.items():
                if None in measurements:
                    recommender_mean_measurements[measure] = None
                else:
                    recommender_mean_measurements[measure] = np.mean(measurements)
            recommenders_mean_measurements.append(recommender_mean_measurements)

        return recommenders_mean_measurements

    def get_raw_results(self):
        """
        Returns the raw results of the benchmark evaluation.

        The raw results are a list of tuples, where each tuple contains the parameters used
        for evaluation and a dictionary of measurements for each recommender. If a measurement
        is a NumPy array, it is converted to a list before being included in the dictionary.

        Returns:
            list: A list of tuples containing the parameters and measurements for each recommender.
        """
        return [
            (
                params,
                {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in measure.items()},
            )
            for params, measure in zip(self.param_combinations, self.recommenders_measurements)
        ]

    def get_ordering_results(self):
        """
        Returns the ordering results for each measure.

        The ordering results are a dictionary where the keys are the measure names and the values
        are lists of sorted solution IDs. The sorting is based on the measure values obtained from
        the recommender mean measurements. Lists are always sorted in ascending order of utility.
        Therefore, if a measure is better when lower, the list is reversed. In other words, the
        last element of the list is always the best solution. If there are None values in any of the
        measurements, the corresponding solution is excluded from the ordering but the numbering is
        preserved according to the raw measurements.

        Returns:
            dict: A dictionary containing the ordering results for each measure.
        """
        ordering = {}

        measures = self.train_measures + self.test_measures

        for measure in measures:
            measure_name = measure.get_name()

            ids_from_valid_solutions = []
            valid_solutions = []

            for result_id, recommender_mean_measurements in enumerate(
                self.recommenders_mean_measurements
            ):

                is_solution_valid = all(
                    [
                        mean_measure is not None
                        for mean_measure in recommender_mean_measurements.values()
                    ]
                )
                if is_solution_valid:
                    ids_from_valid_solutions.append(result_id)
                    valid_solutions.append(recommender_mean_measurements[measure_name])

            sorted_solutions = np.argsort(valid_solutions)
            sorted_ids = [ids_from_valid_solutions[i] for i in sorted_solutions]

            if measure.is_better_higher():
                ordering[measure_name] = sorted_ids
            else:
                ordering[measure_name] = sorted_ids[::-1]

        return ordering

    def get_best_results(self):
        """
        Returns a dictionary containing the best results for each measure.

        Returns:
        A dictionary containing the best results for each measure.
        The dictionary contains the following information for each measure (key):
        - parameters: The parameter combination associated with the best measurement.
        - other_metrics: Other metrics associated with the best parameter combination.
        - raw: Raw measurements for the best parameter combination.
        - mean: Mean measurement for the best parameter combination.
        - fit_time: Fit time for the best parameter combination.
        - test_time: Test time for the best parameter combination.
        """
        best = {}

        measures = self.train_measures + self.test_measures

        for measure in measures:
            measure_name = measure.get_name()

            if measure.is_better_higher():
                best_parameter_id = self.ordering[measure_name][-1]
            else:
                best_parameter_id = self.ordering[measure_name][0]

            best[measure_name] = {
                "parameters": self.param_combinations[best_parameter_id],
                "other_metrics": self.recommenders_mean_measurements[best_parameter_id],
                "raw": self.recommenders_measurements[best_parameter_id][measure_name].tolist(),
                "mean": self.recommenders_mean_measurements[best_parameter_id][measure_name],
                "fit_time": self.recommenders_mean_measurements[best_parameter_id]["fit_time"],
                "test_time": self.recommenders_mean_measurements[best_parameter_id]["test_time"],
            }

        return best

    def compute_results(self):
        """
        Compute the results of the hyperparameter search.

        Returns:
        Tuple[dict, dict, List[Tuple[dict, dict]]]
            The best parameters for each measure,
            the ascending ordering of the recommenders for each measure (invalid solutions are
                excluded from the ordering but the numbering is preserved according to the raw
                measurements),
            and the raw measurements for each recommender.
        """

        self.recommenders_mean_measurements = self.compute_mean_measurements()

        self.raw = self.get_raw_results()
        self.ordering = self.get_ordering_results()
        self.best = self.get_best_results()

        return self.best, self.ordering, self.raw


class GridSearch(BaseSearch):

    def __init__(
        self,
        recommender_class: Type[AlgoBase],
        param_grid: dict,
        test_measures: List[TestMeasureStrategy] = [],
        train_measures: List[TrainMeasureStrategy] = [],
        max_workers=1,
    ):

        super().__init__(
            recommender_class=recommender_class,
            param_grid=param_grid,
            test_measures=test_measures,
            train_measures=train_measures,
            max_workers=max_workers,
        )

        self.param_combinations = [
            dict(zip(self.param_grid, v)) for v in product(*self.param_grid.values())
        ]

class RandomizedSearch(BaseSearch):

    def __init__(
        self,
        recommender_class: Type[AlgoBase],
        param_grid: dict,
        test_measures: List[TestMeasureStrategy] = [],
        train_measures: List[TrainMeasureStrategy] = [],
        max_workers=1,
        seed=42,
        iterations=10,
    ):

        super().__init__(
            recommender_class=recommender_class,
            param_grid=param_grid,
            test_measures=test_measures,
            train_measures=train_measures,
            max_workers=max_workers,
        )

        self.iterations = iterations
        self.seed = seed

        rng = np.random.default_rng(self.seed)

        complete_param_grid = [
            dict(zip(self.param_grid, v)) for v in product(*self.param_grid.values())
        ]

        self.param_combinations = rng.choice(complete_param_grid, self.iterations, replace=False)
