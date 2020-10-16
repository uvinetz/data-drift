from collections import Counter, OrderedDict
from typing import Union, Tuple, Optional, List

from scipy.stats import mannwhitneyu, ks_2samp, chisquare
import numpy as np
import pandas as pd


class DistributionDrift:
    def __init__(self, significance: float = 0.05):
        """
        Class used for detecting data drift based on distribution evaluations
        """
        self.significance = significance

    def _compare_two_numerical_distributions(
        self,
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
        test: Optional[str] = "mw",
    ) -> bool:
        """
        Compares two numerical distributions based on the selected test
        """
        self._validate_test_name(test)
        test_dict = dict(mw=mannwhitneyu, ks=ks_2samp)
        statistic, p_value = test_dict[test](baseline, new, alternative="two_sided")
        return p_value < self.significance

    def _compare_two_categorical_distributions(
        self,
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
    ) -> bool:
        """
        Compares two categorical distributions based on the chi-squared test
        """
        baseline = [cat for cat in baseline if cat in new]
        new = [cat for cat in new if cat in baseline]
        base_freq, new_freq = self._create_frequency_arrays(baseline, new)
        statistic, p_value = chisquare(new_freq, base_freq)
        return p_value < self.significance

    @staticmethod
    def _validate_test_name(test: str):
        if test not in ["mw", "ks"]:
            raise ValueError("Not a valid test name")

    @staticmethod
    def test_new_categories(
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
    ) -> list:
        return [cat for cat in new if cat not in baseline]

    @staticmethod
    def test_deprecated_categories(
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
    ) -> list:
        return [cat for cat in baseline if cat not in new]

    @staticmethod
    def _create_frequency_arrays(
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
    ) -> Tuple[List[int], List[int]]:
        """
        This method creates frequency arrays from the original arrays, i.e. arrays with the frequency of each category.
        """

        base_cat_freq = OrderedDict(
            {key: freq for key, freq in Counter(baseline).items() if key in new}
        )
        new_cat_freq = OrderedDict(
            {key: Counter(new)[key] for key in base_cat_freq.keys()}
        )

        base_freq = list(base_cat_freq.values())
        new_freq = list(new_cat_freq.values())

        return base_freq, new_freq

    def detect_drift(
        self,
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
        numerical_test: Optional[str] = "mw",
        is_categorical: Optional[bool] = False,
    ) -> bool:
        if is_categorical:
            return self._compare_two_categorical_distributions(baseline, new)
        return self._compare_two_numerical_distributions(
            baseline, new, test=numerical_test
        )
