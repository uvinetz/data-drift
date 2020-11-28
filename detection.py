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

    def _create_frequency_arrays(
        self,
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

        base_cat_freq, new_cat_freq = self._remove_missing_values(baseline, new, base_cat_freq, new_cat_freq)

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
        """
        :param baseline: Baseline (old period / train set) vector
        :param new: Target (future period / test set) vector
        :param numerical_test: Which statistical test to use if the variable is numerical
        :param is_categorical: Whether the variable is categorical or not.
        :return: True if a drift was detected between baseline and new.
        """
        if is_categorical:
            return self._compare_two_categorical_distributions(baseline, new)
        return self._compare_two_numerical_distributions(
            baseline, new, test=numerical_test
        )

    def _remove_missing_values(
        self,
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
        base_cat_freq: OrderedDict,
        new_cat_freq: OrderedDict
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        This method gets the missing values frequencies from the original arrays and compares missing values frequencies
        :param baseline: Baseline (old period / train set) vector
        :param new: Target (future period / test set) vector
        :param base_cat_freq: Baseline vector value counts with missing values
        :param new_cat_freq: Target vector value counts with missing values
        """
        # Transform inputs into pandas series
        baseline = pd.Series(baseline)
        new = pd.Series(new)

        # Calculate proportions of missing values
        base_missing_prop = baseline.isna().mean()
        new_missing_prop = new.isna().mean()

        # Validate missing values proportions
        if base_missing_prop != 0 and new_missing_prop != 0 and \
                (new_missing_prop > base_missing_prop * (1 + self.significance) or
                 base_missing_prop > new_missing_prop * (1 + self.significance)):
            raise ValueError("Too many missing values")
        elif base_missing_prop == 0 and new_missing_prop > self.significance:
            raise ValueError("Too many missing values")
        elif new_missing_prop == 0 and base_missing_prop > self.significance:
            raise ValueError("Too many missing values")

        # Remove the missing values
        base_cat_freq = OrderedDict({key: value for key, value in base_cat_freq.items() if key})
        new_cat_freq = OrderedDict({key: value for key, value in new_cat_freq.items() if key})

        return base_cat_freq, new_cat_freq
