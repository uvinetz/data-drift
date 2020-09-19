import pandas as pd


class BaseDetector:
    def __init__(self, param):
        self._param = param


class TimeBasedDetector(BaseDetector):
    def __init__(self, time_range: str):
        if time_range not in ["day", "month", "year"]:
            raise ValueError("Not a valid range")
        self._time_range = time_range


class SampleBasedDetector(BaseDetector):
    def __init__(self, param):
        self._param = param
