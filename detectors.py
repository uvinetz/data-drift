import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp


class BaseDetector:
    def compare_two_numerical_distributions(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        test: str = "mw",
        significance: float = 0.05,
    ):
        self._validate_test_name(test)
        test_dict = dict(mw=mannwhitneyu, ks=ks_2samp)
        statistic, p_value = test_dict[test](x1, x2, alternative="two_sided")
        return p_value < significance

    @staticmethod
    def _validate_test_name(test):
        if test not in ["mw", "ks"]:
            raise ValueError("Not a valid test name")


class TimeBasedDetector(BaseDetector):
    def __init__(self, time_range: str):
        if time_range not in ["day", "month", "year"]:
            raise ValueError("Not a valid range")
        self._time_range = time_range


class SampleBasedDetector(BaseDetector):
    def __init__(self, param):
        self._param = param
