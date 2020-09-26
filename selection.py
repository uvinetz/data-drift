import pandas as pd


class TimeBasedSelector:
    def __init__(self, time_range: str):
        if time_range not in ["day", "month", "year"]:
            raise ValueError("Not a valid range")
        self._time_range = time_range


class ColumnBasedSelector:
    def __init__(self, param):
        self._param = param
