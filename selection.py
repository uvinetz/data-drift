from typing import List, Tuple, Any, Union
from datetime import datetime as dt

import pandas as pd


class TimeBasedSelector:
    def __init__(
        self,
        time_cutoffs: List[dt],
        value_column_names: Union[Any, List[Any]],
        splitting_column_name: Any,
    ):
        """
        Class to split a pandas dataframe by datetime chunks
        :param time_cutoffs: dates selected to slice the data. Each date/timestamp marks the end of the period.
        """
        if not isinstance(time_cutoffs, list):
            raise TypeError("time_ranges should be a list")
        if not time_cutoffs:
            raise ValueError("time_ranges should contain at least one date/timestamp")
        if not self._check_cutoffs_order(time_cutoffs):
            raise ValueError("The order in the list is not correct")
        self._time_ranges = time_cutoffs
        self._columns = value_column_names
        self._splitter = splitting_column_name

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

    def split_dataframe(
        self, df: pd.DataFrame, time_cutoffs: List[dt]
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        list_of_splits: List[pd.DataFrame] = list()
        temp_df = df.copy()
        for cutoff in time_cutoffs:
            list_of_splits.append(temp_df[temp_df[self._splitter] < cutoff])
            temp_df = temp_df[temp_df[self._splitter] >= cutoff]
        return [
            (list_of_splits[ix], list_of_splits[ix + 1])
            for ix in range(len(list_of_splits) - 1)
        ]


class ColumnBasedSelector:
    def __init__(self, param):
        self._param = param
