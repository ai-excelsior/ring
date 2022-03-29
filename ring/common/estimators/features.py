from __future__ import annotations

from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.tseries.frequencies import to_offset

from .base import Estimator


class DatetimeComponentsFeatureExtractor(Estimator):
    """
    Extract datetime components from datetime column.
    """

    __supported_components__ = {
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "day_of_week",
        "day_of_year",
    }

    __one_to_zero__ = {
        "month",
        "day",
        "day_of_year",
    }

    def __init__(self, time_column: str, features: List[str], method="zscore") -> None:
        """
        method can be one of zscore or embedding

        when embedding method, we do nothing
        """
        super().__init__()

        assert all(
            feature in self.__supported_components__ for feature in features
        ), f"All features should in the {self.__supported_components__}"

        self._features = features
        self._time_column = time_column
        self._method = method

    def _get_time_serie(self, data: DataFrame):
        return data.index.to_series() if self._time_column == "index" else data[self._time_column]

    def fit(self, data: DataFrame):
        assert all(
            [feature_name not in data.columns for feature_name in self._features]
        ), f"The {self._features} is preseved, but it already exist in data."

        state = dict()
        time_serie = self._get_time_serie(data)
        for feature_name in self._features:
            feature_column: Series = getattr(time_serie.dt, feature_name)
            if self._method == "zscore":
                state[feature_name] = (
                    feature_column.mean(),
                    feature_column.std(),
                )

        self._state = state

    def transform(self, data: DataFrame):
        self._assert_fitted()

        time_serie = self._get_time_serie(data)
        for feature_name in self._features:
            feature_column: Series = getattr(time_serie.dt, feature_name)
            if feature_name in self.__one_to_zero__:
                feature_column -= 1

            if self._method == "zscore":
                mean, std = self._state[feature_name]
                data = data.assign(**{feature_name: (feature_column - mean) / std})
            elif self._method == "embedding":
                data = data.assign(**{feature_name: feature_column})

        return data


class DatetimeBooleanFeatureExtractor(Estimator):
    __supported_components__ = {
        "is_year_start",
        "is_year_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_month_start",
        "is_month_end",
        "is_weekend",
    }

    __map_components_to_bool__ = {
        "is_weekend": lambda x: x.day_of_week in (5, 6),
    }

    def __init__(self, time_column: str, features: List[str]) -> None:
        super().__init__()

        assert all(
            feature in self.__supported_components__ for feature in features
        ), f"All features should in the {self.__supported_components__}"

        self._time_column = time_column
        self._features = features

    @classmethod
    def get_column_names(cls, time_column: str, feature: str):
        return f"_{time_column}_{feature}_"

    def _get_time_serie(self, data: DataFrame):
        return data.index.to_series() if self._time_column == "index" else data[self._time_column]

    def fit(self, _: DataFrame):
        self._state = [
            self.get_column_names(self._time_column, feature_name) for feature_name in self._features
        ]

    def transform(self, data: DataFrame):
        self._assert_fitted()

        time_serie: Series = self._get_time_serie(data)
        apply_calls = set(self.__map_components_to_bool__.keys())
        for feature_name in self._features:
            column_name = self.get_column_names(self._time_column, feature_name)
            if feature_name in apply_calls:
                data[column_name] = time_serie.apply(self.__map_components_to_bool__[feature_name])
            else:
                data[column_name] = getattr(time_serie.dt, feature_name)
        return data


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


class GlobalTimeIndexExtractor(Estimator):
    """
    add _time_idx_ to dataframe
    """

    def __init__(self, freq: str = None) -> None:
        """
        Args:
            freq (str, optional): pandas freq str. Defaults will infer from pandas.
        """
        super().__init__()
        self._state = "NotRequired"
        self._freq = freq

    def transform(self, data: pd.Series):
        self._assert_fitted()
        # drop the time zone info
        data = pd.to_datetime(data).dt.tz_localize(None)
        freq = self._freq or pd.infer_freq(data)
        assert freq is not None, "Auto frequency infer failed, please provide it manully."

        offset = to_offset(freq)
        start_time = pd.Timestamp(data.min().value)
        # TODO: more offset should be supported in here.
        # MS: Month Start
        if freq == "MS":
            return data.apply(lambda x: diff_month(x, start_time))
        # TODO: perform this operator by each group
        return ((data - start_time) / offset).astype(int)


def get_first_none_zero(data: Series):
    for index, value in data.iteritems():
        if not pd.isna(value) and value != 0:
            return index
    return data.index[0]


def get_last_none_zero(data: Series):
    for index, row in data[::-1].iteritems():
        if not pd.isna(row) and row != 0:
            return index
    return data.index[-1]


def group_filter_by_rules(data: DataFrameGroupBy, rules: Set[str]):
    start_pos = data.index[0]
    end_pos = data.index[-1]

    if "start_with_none_zero" in rules:
        start_pos = get_first_none_zero(data)

    if "end_with_none_zero" in rules:
        end_pos = get_last_none_zero(data)

    return np.logical_and(data.index >= start_pos, data.index <= end_pos)


class MaskExtractor(Estimator):
    """
    Extract time sequence mask from the dataframe based on mask the mask config.
    """

    def __init__(self, mask_config: Dict[str, List[str]], group_ids: List[str] = []):
        """
        time_idx: indicate the order of sequence.
        mask_config: {[column_name]: [mask_rules]}, mask {column_name} by mask_rules.
            Each rule will concat with `and` operation.
            mask_rules support start_with_none_zero|end_with_none_zero
        group_ids: how to group the data
        """
        super().__init__()
        self._mask_config = mask_config
        self._group_ids = group_ids
        self._state = "NotRequired"

    @classmethod
    def get_masked_column_name(cls, column_name: str):
        return f"_{column_name}_mask_"

    def fit(self, _: DataFrame):
        pass

    def transform(self, data: DataFrame):
        self._assert_fitted()
        columns = set(data.columns)
        assert all(
            [mask_column_name in columns for mask_column_name in self._mask_config.keys()]
        ), "All mask_config.keys() should exist in passed-in dataframe"
        assert all(
            [group_id in columns for group_id in self._group_ids]
        ), "All group_ids should exist in passed-in dataframe"

        for column_name in self._mask_config.keys():
            mask_rules = set(self._mask_config[column_name])
            target = self.get_masked_column_name(column_name)
            if len(self._group_ids) > 0:
                groupped = data.groupby(self._group_ids)
                data[target] = groupped[column_name].transform(lambda x: group_filter_by_rules(x, mask_rules))
            else:
                data[target] = group_filter_by_rules(data[column_name], mask_rules)

        return data


class BooleanMerger(Estimator):
    """
    Merge boolean columns
    """

    ALG_MAP = {"and": "&", "or": "|"}

    def __init__(self, source: Union[str, List[str]], target: str, alg="and") -> None:
        super().__init__()
        self._source = [source] if isinstance(source, str) else source
        self._target = target
        self._alg = alg
        self._state = "NotRequired"

    def fit(self, _: DataFrame):
        pass

    def transform(self, data: DataFrame):
        self._assert_fitted()
        columns = set(data.columns)
        assert (
            self._target not in columns
        ), f"The passed-in data frame should not have column `{self._target}`"

        # do validation
        assert all(
            [source_column_name in columns for source_column_name in self._source]
        ), f"The passed-in data frame don't have all columns, `{self._source}`"

        expr = f"{self._target}={self.ALG_MAP[self._alg].join(self._source)}"

        data = data.eval(expr)
        data.drop(self._source, axis=1, inplace=True)

        return data


class BooleanFilter(Estimator):
    """
    Filter columns which value is True.
    """

    def __init__(self, by: str) -> None:
        super().__init__()
        self._by = by
        self._state = "NotRequired"

    def fit(self, _):
        pass

    def transform(self, data: DataFrame):
        self._assert_fitted()

        data = data[data[self._by]]
        return data.drop(self._by, axis=1)


class CategoricalEncoder(Estimator):
    """
    Encode categoricals to int, always encode nan and unknown classes (in transform) as class `0` if exists
    """

    def __init__(self):
        """
        init CategoricalEncoder

        Args:
            add_nan: if to force encoding of nan at 0
        """
        super().__init__()

    @staticmethod
    def is_numeric(data: pd.Series) -> bool:
        """
        Determine if series is numeric or not.
        """
        return data.dtype.kind in "bcif" or (
            isinstance(data, pd.CategoricalDtype) and data.cat.categories.dtype.kind in "bcif"
        )

    def fit(self, data: pd.Series):
        self._state = ["<UKN>"]
        self._state.extend(data.astype("category").cat.categories.to_list())

    def transform(self, data: pd.Series):
        self._assert_fitted()

        value_index_map = {v: i for i, v in enumerate(self._state)}
        return data.apply(lambda x: value_index_map.get(x, 0), convert_dtype=False).astype("i")

    def inverse_transform(self, data: pd.Series):
        self._assert_fitted()

        index_value_map = {i: v for i, v in enumerate(self._state)}

        return data.map(index_value_map).astype("i")


class SimpleMovingAverage(Estimator):
    """
    A simple moving average, may not useful at all. When processing, we assume that the data is already sorted.

    if i < window_size, keep the existing data
    else the nth value will be (x0 + x1 + ... + xn) / window_size
    """

    def __init__(self, window_size: int, group_ids: List[str] = []) -> None:
        super().__init__()
        self._window_size = window_size
        self._group_ids = group_ids
        self._state = "NotRequired"

    def _sma_transform(self, serie: pd.Series, window_size: int):
        if len(serie) <= window_size:
            raise ValueError("window size is too small to do a exponential moving average transform")

        # calc simple moving average
        for i in reversed(range(window_size - 1, len(serie))):
            serie.iloc[i] = serie.iloc[i - window_size + 1 : i + 1].mean()

        return serie

    def transform(self, data: pd.Series, source: pd.DataFrame = None):
        if len(self._group_ids) > 0 and source is None:
            raise ValueError("Must supply `source` to transform, in order to group by")

        # without group_ids
        if len(self._group_ids) == 0:
            return self._sma_transform(data.copy(deep=True), self._window_size)

        # with group_ids
        return (
            source[self._group_ids]
            .assign(y=data)
            .groupby(self._group_ids, observed=True)["y"]
            .transform(lambda x: self._sma_transform(x.copy(deep=True), self._window_size))
        )

    # def _inverse_serie(self, serie: pd.Series, window_size: int):
    #     for i in range(len(serie) - window_size + 1):
    #         target_position = i + window_size - 1
    #         target_value = serie.iloc[target_position]
    #         serie.iloc[target_position] = target_value * window_size - serie.iloc[i:target_position].sum()
    #     return serie

    # def inverse_transform(self, data: pd.DataFrame):
    #     # without group_ids
    #     if len(self._group_ids) == 0:
    #         for column_name, rolling_window_size in self._config.items():
    #             self._inverse_serie(data[column_name], rolling_window_size)
    #         return data

    #     # with group_ids
    #     group = data.groupby(self._group_ids)
    #     for column_name, rolling_window_size in self._config.items():
    #         for idx in group.indices.values():
    #             column_index = data.columns.get_loc(column_name)
    #             data.iloc[idx, column_index] = self._inverse_serie(
    #                 data.iloc[idx, column_index], rolling_window_size
    #             )

    #     return data


class ExponentialMovingAverage(Estimator):
    """
    A exponential moving average. When processing, we assume that the data is already sorted.

    EMA(i) = Value(i) * alpha + (1 - alpha) * EMA(i-1)

    if i < window_size, keep the existing data in order to output the same size.
    """

    def __init__(self, window_size: int = 1, alpha: float = None, group_ids: List[str] = []) -> None:
        super().__init__()
        self._window_size = window_size
        self._alpha = alpha
        self._group_ids = group_ids
        self._state = "NotRequired"

        if self._alpha is None:
            self._alpha = (window_size - 1.0) / (window_size + 1.0)

    def _ema_transform(self, serie: pd.Series, window_size: int, alpha: int):
        if len(serie) <= window_size:
            raise ValueError("window size is too small to do a exponential moving average transform")

        # using simple average to fill the last value in the first window
        sma = serie[:window_size].sum() / window_size
        serie.iloc[window_size - 1] = sma

        # calc exponential moving average
        for i in range(window_size, len(serie)):
            serie.iloc[i] = serie.iloc[i] * alpha + (1 - alpha) * serie.iloc[i - 1]

        return serie

    def transform(self, data: pd.Series, source: pd.DataFrame = None):
        if len(self._group_ids) > 0 and source is None:
            raise ValueError("Must supply `source` to transform, in order to group by")

        # without group_ids
        if len(self._group_ids) == 0:
            return self._ema_transform(data.copy(deep=True), self._window_size, self._alpha)

        # with group_ids
        return (
            source[self._group_ids]
            .assign(y=data)
            .groupby(self._group_ids, observed=True)["y"]
            .transform(lambda x: self._ema_transform(x.copy(deep=True), self._window_size, self._alpha))
        )


class WeightedMovingAverage(Estimator):
    """
    A weighted moving average. When processing, we assume that the data is already sorted.

    WMA(i) = WMA(i)*n + WMA(i-1)*(n-1) + WMA(i-2)*(n-2) + ... + WMA(i-n)*(1)
        with i, i-th WMA value
        with n, window_size

    if i < window_size, keep the existing data in order to output the same size.
    """

    def __init__(self, window_size: int = 1, group_ids: List[str] = []) -> None:
        super().__init__()
        self._window_size = window_size
        self._group_ids = group_ids
        self._state = "NotRequired"

    @property
    def weights(self):
        weights = np.array(range(1, 1 + self._window_size))
        return weights / weights.sum()

    def _wma_transform(self, serie: Series, window_size: int):
        if len(serie) <= window_size:
            raise ValueError("window size is too small to do a weighted moving average transform")

        # calc exponential moving average
        weights = self.weights
        for i in reversed(range(window_size, len(serie))):
            serie.iloc[i] = (serie.iloc[i + 1 - window_size : i + 1] * weights).sum()

        return serie

    def transform(self, data: pd.Series, source: pd.DataFrame = None):
        if len(self._group_ids) > 0 and source is None:
            raise ValueError("Must supply `source` to transform, in order to group by")

        # without group_ids
        if len(self._group_ids) == 0:
            return self._wma_transform(data.copy(deep=True), self._window_size)

        # with group_ids
        return (
            source[self._group_ids]
            .assign(y=data)
            .groupby(self._group_ids, observed=True)["y"]
            .transform(lambda x: self._wma_transform(x.copy(deep=True), self._window_size))
        )


if __name__ == "__main__":
    pass
