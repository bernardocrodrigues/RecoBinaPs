import logging

DEFAULT_LOGGER = logging.getLogger("FormalConceptAnalysis")
DEFAULT_LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch.setFormatter(formatter)
DEFAULT_LOGGER.addHandler(ch)
