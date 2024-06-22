import logging

import numpy as np
import numba as nb

from .common import (
    _cosine_similarity,
    _adjusted_cosine_similarity,
    _user_pattern_similarity,
    _get_similarity_matrix,
)

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


DEFAULT_LOGGER.info("Forcing Numba compilation...")

# u_float = np.array([1, 2, 3, 4, 5], dtype=np.float64)
# v_float = np.array([1, 2, 3, 4, 5], dtype=np.float64)
# u_int = np.array([1, 2, 3, 4, 5], dtype=np.int64)
# v_int = np.array([1, 2, 3, 4, 5], dtype=np.int64)

# dataset_float = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.float64)
# dataset_int = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.int64)

# _cosine_similarity(u_float, v_float, 1e-7)
# _cosine_similarity(dataset_float[0], dataset_float[1])

# _adjusted_cosine_similarity(u_float, v_float)
# _adjusted_cosine_similarity(dataset_float[0], dataset_float[1])

# _user_pattern_similarity(u_int, v_int)
# _user_pattern_similarity(dataset_int[0], dataset_int[1])

# _get_similarity_matrix(dataset_float)


DEFAULT_LOGGER.info("Forcing Numba compilation OK")
