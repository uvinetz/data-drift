from typing import List
from datetime import datetime as dt

import pandas as pd


class TimeBasedSelector:
    def __init__(self, time_cutoffs: List[dt]):
        """
        Class to split a pandas dataframe by datetime chunks
        :param time_cutoffs: dates selected to slice the data. Each date/timestamp marks the end of the period.
        :type time_cutoffs: list
        """
        if not isinstance(time_cutoffs, list):
            raise TypeError("time_ranges should be a list")
        if not time_cutoffs:
            raise ValueError("time_ranges should contain at least one date/timestamp")
        if not self._check_cutoffs_order(time_cutoffs):
            raise ValueError("The order in the list is not correct")
        self._time_ranges = time_cutoffs

    @staticmethod
    def _check_cutoffs_order(time_cutoffs: List[dt]) -> bool:
        """
        This function checks if the order of the cutoffs (when two or more present) is correct (lower to higher)
        """
        bad_cutoffs = [
            ix
            for ix in range(1, len(time_cutoffs))
            if time_cutoffs[ix] <= time_cutoffs[ix - 1]
        ]
        return not bad_cutoffs


class ColumnBasedSelector:
    def __init__(self, param):
        self._param = param
