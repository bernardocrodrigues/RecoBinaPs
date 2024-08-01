import logging


def build_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

DEFAULT_LOGGER = build_logger("recommenders", logging.INFO)
DEBUG_LOGGER = build_logger("recommenders debug", logging.DEBUG)
