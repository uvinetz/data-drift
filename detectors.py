from scipy.stats import mannwhitneyu, ks_2samp


class BaseDetector:
    def compare_two_numerical_distributions(
        self, baseline, new, test="mw", significance=0.05,
    ):
        self._validate_test_name(test)
        test_dict = dict(mw=mannwhitneyu, ks=ks_2samp)
        statistic, p_value = test_dict[test](baseline, new, alternative="two_sided")
        return p_value < significance

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
        :param baseline:
        :param new:
        :return:
        """
        pass


class TimeBasedDetector(BaseDetector):
    def __init__(self, time_range: str):
        if time_range not in ["day", "month", "year"]:
            raise ValueError("Not a valid range")
        self._time_range = time_range


class SampleBasedDetector(BaseDetector):
    def __init__(self, param):
        self._param = param
