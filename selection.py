import pandas as pd
from datetime import datetime as dt


class TimeBasedSelector:
    def __init__(self, time_cutoffs, datetime_format):
        """
        Class to split a pandas dataframe by datetime chunks
        :param time_cutoffs: dates selected to slice the data. Each date/timestamp marks the end of the period.
        :type time_cutoffs: list
        """
        if not isinstance(time_cutoffs, list):
            raise TypeError("time_ranges should be a list")
        if len(time_cutoffs) < 1:
            raise ValueError("time_ranges should contain at least one date/timestamp")
        if len(time_cutoffs) > 1:
            if not self._check_cutoffs_order(time_cutoffs):
                raise ValueError("The order in the list is not correct")
        self._time_ranges = time_cutoffs

    @staticmethod
    def _check_cutoffs_order(time_cutoffs):
        """
        This function checks if the order of the cutoffs (when two or more present) is correct (lower to higher)
        :param time_cutoffs:
        :rtype: bool
        """
        return True


class ColumnBasedSelector:
    def __init__(self, param):
        self._param = param
