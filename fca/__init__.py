import logging

DEFAULTLOGGER = logging.getLogger("FormalConceptAnalysis")
DEFAULTLOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch.setFormatter(formatter)
DEFAULTLOGGER.addHandler(ch)

from .formal_concept_analysis import *