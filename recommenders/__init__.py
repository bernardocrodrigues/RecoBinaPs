import logging

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
