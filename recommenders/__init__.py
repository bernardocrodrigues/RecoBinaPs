import logging

import numpy as np

from .common import (
    cosine_similarity,
    adjusted_cosine_similarity,
    user_pattern_similarity,
    weight_frequency,
    get_similarity,
    get_similarity_matrix,
    get_top_k_biclusters_for_user,
    get_indices_above_threshold,
    merge_biclusters,
)

from pattern_mining.formal_concept_analysis import create_concept


def compile_numba():
    DEFAULT_LOGGER.info("Forcing Numba compilation...")

    u_float = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    v_float = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    u_int = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    concept = create_concept([1, 2, 3], [1, 2, 3])

    dataset_float = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.float64)
    dataset_int = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.int64)

    cosine_similarity(u_float, v_float, 1e-7)
    cosine_similarity(dataset_float[0], dataset_float[1])

    adjusted_cosine_similarity(u_float, v_float)
    adjusted_cosine_similarity(dataset_float[0], dataset_float[1])

    user_pattern_similarity(u_int, concept)
    user_pattern_similarity(dataset_int[0], concept)
    
    weight_frequency(dataset_int[0], concept)

    get_similarity(0, 0, dataset_float)

    get_similarity_matrix(dataset_float)

    get_top_k_biclusters_for_user([concept], u_int, 1)

    get_indices_above_threshold(u_int, 2)

    merge_biclusters([concept, concept])

    DEFAULT_LOGGER.info("Forcing Numba compilation OK")


DEFAULT_LOGGER = logging.getLogger("recommenders")
DEFAULT_LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch.setFormatter(formatter)
DEFAULT_LOGGER.addHandler(ch)

DEBUG_LOGGER = logging.getLogger("recommenders debug")
DEBUG_LOGGER.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch.setFormatter(formatter)
DEBUG_LOGGER.addHandler(ch)

compile_numba()
