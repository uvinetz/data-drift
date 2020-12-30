from collections import Counter, OrderedDict
from typing import Union, Tuple, Optional, List, Any, Dict
from datetime import datetime as dt

from scipy.stats import mannwhitneyu, ks_2samp, power_divergence
import numpy as np
import pandas as pd

from selection import TimeBasedSelector


class DistributionDrift:
    def __init__(self, significance: float = 0.05):
        """
        Class used for detecting data drift based on distribution evaluations
        :param significance: Statistical significance required in the different tests to flag drift
        """
        self._significance: float = significance
        self.drift_alerts: List[Dict[str, str]] = list()
        self.drift_alerts_count: int = 0

    def _compare_two_numerical_distributions(
        self,
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
        test: Optional[str] = "ks",
    ) -> bool:
        """
        Compares two numerical distributions based on the selected test
        """
        self._validate_test_name(test, "numerical")
        test_dict = dict(mw=mannwhitneyu, ks=ks_2samp)
        statistic, p_value = test_dict[test](baseline, new, alternative="two-sided")
        return p_value < self._significance

    def _compare_two_categorical_distributions(
        self,
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
        test: Optional[str] = "pearson",
    ) -> bool:
        """
        Compares two categorical distributions based on the chi-squared test
        """
        self._validate_test_name(test, "categorical")
        baseline = [cat for cat in baseline if cat in new]
        new = [cat for cat in new if cat in baseline]
        base_freq, new_freq = self._create_frequency_arrays(baseline, new)
        statistic, p_value = power_divergence(new_freq, base_freq, lambda_=test)
        return p_value < self._significance

    @staticmethod
    def _validate_test_name(test: str, variable_type: str):
        if variable_type == "numerical":
            if test not in ["mw", "ks"]:
                raise ValueError("Not a valid test name")
        elif variable_type == "categorical":
            if test not in [
                "pearson",
                "log-likelihood",
                "freeman-tukey",
                "mod-log-likelihood",
                "neyman",
                "cressie-read",
            ]:
                raise ValueError("Not a valid test name")

    @staticmethod
    def _test_new_categories(
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
    ) -> list:
        return [cat for cat in new if cat not in baseline]

    @staticmethod
    def _test_deprecated_categories(
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

    def detect_single_drift(
        self,
        baseline: Union[list, np.ndarray, pd.Series],
        new: Union[list, np.ndarray, pd.Series],
        numerical_test: Optional[str] = "ks",
        is_categorical: Optional[bool] = False,
        categorical_test: Optional[str] = "pearson",
    ) -> bool:
        """
        :param baseline: Baseline (old period / train set) vector
        :param new: Target (future period / test set) vector
        :param numerical_test: Which statistical test to use if the variable is numerical
        :param is_categorical: Whether the variable is categorical or not.
        :param categorical_test: Which statistical test to use if the variable is categorical
        :return: True if a drift was detected between baseline and new.
        """
        if is_categorical:
            return self._compare_two_categorical_distributions(
                baseline, new, categorical_test
            )
        return self._compare_two_numerical_distributions(
            baseline, new, test=numerical_test
        )

    def detect_drift(
        self,
        df: pd.DataFrame,
        feature_names: Union[Any, List[Any]],
        splitting_column_name: Any,
        time_cutoffs: List[dt],
        categorical_columns: Optional[List[str]] = None,
        numerical_test: Optional[str] = "ks",
        categorical_test: Optional[str] = "pearson",
    ):
        selector = TimeBasedSelector(time_cutoffs, feature_names, splitting_column_name)
        list_of_splits = selector.split_dataframe(df)
        for ix, split in enumerate(list_of_splits):
            for feature in feature_names:
                baseline, new = split[0][feature], split[1][feature]
                if self.detect_single_drift(
                    baseline,
                    new,
                    numerical_test,
                    is_categorical=(feature in categorical_columns),
                    categorical_test=categorical_test,
                ):
                    print(
                        f"Drift detected in feature {feature} before and after {time_cutoffs[ix]}"
                    )
                    self.drift_alerts.append(
                        dict(feature=feature, time_cutoff=time_cutoffs[ix])
                    )
        self.drift_alerts_count = len(self.drift_alerts)
