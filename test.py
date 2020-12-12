import time
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytest

from selection import TimeBasedSelector

np.random.seed(99)


def str_time_prop(start: str, end: str, string_format: str, prop: float):
    start_time = time.mktime(time.strptime(start, string_format))
    end_time = time.mktime(time.strptime(end, string_format))
    random_time = start_time + prop * (end_time - start_time)
    return time.strftime(string_format, time.localtime(random_time))


def random_dates(start: str, end: str, size: int = 10000):
    return [
        str_time_prop(start, end, "%Y-%m-%d", np.random.random()) for _ in range(size)
    ]


def generate_synthetic_data(size: int = 10000):
    normal_t1 = np.random.normal(0, 1, size)
    normal_t2 = np.random.normal(0, 1.01, size)
    normal_t3 = np.random.normal(0, 2, size)
    normal_t4 = np.random.normal(1, 2, size)
    normal = np.concatenate([normal_t1, normal_t2, normal_t3, normal_t4])

    categorical_t1 = np.random.choice([0, 1], size, p=[0.8, 0.2])
    categorical_t2 = np.random.choice([0, 1], size, p=[0.801, 0.199])
    categorical_t3 = np.random.choice([0, 1], size, p=[0.7, 0.3])
    categorical_t4 = np.random.choice([0, 1, 2], size, p=[0.5, 0.3, 0.2])
    categorical = np.concatenate(
        [categorical_t1, categorical_t2, categorical_t3, categorical_t4]
    )

    dates_t1 = random_dates("2017-01-01", "2018-01-01", size)
    dates_t2 = random_dates("2018-01-01", "2019-01-01", size)
    dates_t3 = random_dates("2019-01-01", "2020-01-01", size)
    dates_t4 = random_dates("2020-01-01", "2021-01-01", size)
    dates = dates_t1 + dates_t2 + dates_t3 + dates_t4

    df = pd.DataFrame({"date": dates, "numerical": normal, "categorical": categorical})
    df = df.sample(frac=1)
    return df


@pytest.mark.cutoffs
def test_cutoffs():
    df = generate_synthetic_data()
    splitter = TimeBasedSelector(
        [
            dt.strptime("2018-01-01", "%Y-%m-%d"),
            dt.strptime("2019-01-01", "%Y-%m-%d"),
            dt.strptime("2020-01-01", "%Y-%m-%d"),
            dt.strptime("2021-01-01", "%Y-%m-%d"),
        ],
        ["numerical", "categorical"],
        "date",
    )
    splits = splitter.split_dataframe(df)
    assert len(splits) == 3, "Bad number of splits"
