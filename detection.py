from scipy.stats import mannwhitneyu, ks_2samp, chisquare


class DistributionDrift:
    def __init__(self, significance=0.05):
        """
        Class used for detecting data drift based on distribution evaluations
        :param significance: Level of statistical significance of the different tests needed to alert on drift
        :type significance: float, optional
        """
        self.significance = significance

    def _compare_two_numerical_distributions(
        self, baseline, new, test="mw",
    ):
        """
        Compares two numerical distributions based on the selected test
        :param baseline: original values
        :type baseline: array-like
        :param new: new values
        :type new: array-like
        :param test: Statistical test used to evaluate drift. mw: Mann-Whitney and ks: Kolmogorov-Smirnov are supported
        :type test: str, optional
        :return: Whether the distribution drift is significant
        :rtype: bool
        """
        self._validate_test_name(test)
        test_dict = dict(mw=mannwhitneyu, ks=ks_2samp)
        statistic, p_value = test_dict[test](baseline, new, alternative="two_sided")
        return p_value < self.significance

    def _compare_two_categorical_distributions(self, baseline, new):
        """
        Compares two categorical distributions based on the chi-squared test
        :param baseline: original values
        :type baseline: array-like
        :param new: new values
        :type new: array-like
        :return: Whether the distribution drift is significant
        :rtype: bool
        """
        base_freq, new_freq = self._create_frequency_arrays(baseline, new)
        statistic, p_value = chisquare(new_freq, base_freq)
        return p_value < self.significance

    @staticmethod
    def _validate_test_name(test):
        if test not in ["mw", "ks"]:
            raise ValueError("Not a valid test name")

    @staticmethod
    def _test_new_categories(baseline, new):
        return [cat for cat in new if cat not in baseline]

    @staticmethod
    def _test_deprecated_categories(baseline, new):
        return [cat for cat in baseline if cat not in new]

    @staticmethod
    def _create_frequency_arrays(baseline, new):
        """
        This method creates frequency arrays from the original arrays, i.e. arrays with the frequency of each category.
        The order of the categories is preserved between baseline and old. Categories in the new distribution that don't
        exist in the baseline and vice versa should be ignored (a different evaluation is made for that purpose)
        :param baseline: original array of values
        :type baseline: array-like
        :param new: new array of values
        :type new: array-like
        :return: arrays with frequencies for both original and new distributions
        :rtype: tuple of array-like objects
        """

    def detect_drift(self, baseline, new, numerical_test="mw"):
        """
        This uses all the methods above to decide if there is a drift between baseline and new. First, it should check
        that either both arrays are categorical or both numerical. It also has to consider what happens with missing
        values. Then there is a case if both are numerical and another case if both are categorical (the function should
        detect the types, or call a function that does this).
        :param baseline:
        :type baseline: array-like
        :param new:
        :type new: array-like
        :param numerical_test:
        :return: Whether there is drift or not
        :rtype: bool
        """
