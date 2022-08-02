from typing import Dict, Any, List

from .estimators.base import Estimator, AbstractDetrendEstimator, PolynomialDetrendEstimator
import pandas as pd
from .utils import register
import numpy as np
import torch

SEASONALITY: Dict[str, "AbsrtactDetrend"] = {}


def create_detrender_fron_cfg(detrend: bool, group_ids: List[str], targets: List[str]):
    return (
        GroupDetrendTargets(feature_name=targets)
        if detrend and group_ids
        else DetrendTargets(feature_name=targets)
        if detrend
        else AbsrtactDetrend(feature_name=targets)
    )


def serialize_detrender(obj: "AbsrtactDetrend"):
    d = {}
    out = {}
    # for each target
    for k, v in obj._state.items():
        out[k] = (
            [{"k": group_id, "v": estimator.serialize()} for group_id, estimator in v.items()]
            if isinstance(v, dict)
            else v.serialize()
        )
    d["name"] = obj.__class__.__name__
    d["state"] = out
    return d


def deserialize_detrender(d: Dict[str, Any]):
    cls: "AbsrtactDetrend" = SEASONALITY[d["name"]]
    state = d.get("state", {})
    _state = {}
    for k, v in state.items():
        if isinstance(v, list):
            _state[k] = {item["k"]: PolynomialDetrendEstimator.deserialize(item["v"]) for item in v}
        else:
            _state[k] = PolynomialDetrendEstimator.deserialize(v)
    this = cls(feature_name=[k for k, _ in _state.items()])
    this._state = _state
    return this


@register(SEASONALITY)
class AbsrtactDetrend(Estimator):
    def __init__(self, feature_name=None) -> None:
        super().__init__()
        self.feature_name = feature_name

    def fit_transform(self, data: pd.DataFrame, group_ids, **kwargs) -> pd.Series:
        self.fit(data, group_ids)
        return self.transform(data, group_ids, **kwargs)

    def fit(self, data: pd.DataFrame, group_ids):
        self._state = {
            target_column_name: AbstractDetrendEstimator() for target_column_name in self.feature_name
        }
        # for column_name, estimator in self._state.items():
        #     estimator.fit(data[column_name], data["_time_idx_"])
        return self.fit_self(data, group_ids)

    def transform(self, data: pd.DataFrame, group_ids, **kwargs):
        return self.transform_self(data, group_ids, **kwargs)

    def inverse_transform(self, data: pd.DataFrame, group_ids, **kwargs):
        return self.inverse_transform_self(data, group_ids, **kwargs)

    def fit_self(self, data: pd.DataFrame, group_ids):
        pass

    def transform_self(self, data: pd.DataFrame, group_ids, **kwargs):
        return data[self.feature_name]

    def inverse_transform_self(self, data: pd.DataFrame, group_ids, **kwargs):
        return data[self.feature_name]


@register(SEASONALITY)
class DetrendTargets(AbsrtactDetrend):
    def fit_self(self, data: pd.DataFrame, group_ids):
        """
        fit the PolynomialDetrendEstimator, and save it in the state.
        """
        # no group ids
        if len(group_ids) == 0:
            self._state = {
                target_column_name: PolynomialDetrendEstimator() for target_column_name in self.feature_name
            }
            for column_name, estimator in self._state.items():
                estimator.fit(data[column_name], data["_time_idx_"])

    def transform_self(self, data: pd.DataFrame, group_ids):
        assert self._state is not None
        # no group ids
        for column_name, estimator in self._state.items():
            data[column_name] = estimator.transform(data[column_name], data["_time_idx_"])
        return data[self.feature_name]

    def inverse_transform_self(self, data: pd.DataFrame, group_ids):
        assert self._state is not None

        # no group ids
        for column_name, estimator in self._state.items():
            data[column_name] = estimator.inverse_transform(data[column_name], data["_time_idx_"])

        return data[self.feature_name]


@register(SEASONALITY)
class GroupDetrendTargets(AbsrtactDetrend):
    def fit_self(self, data: pd.DataFrame, group_ids):
        # with group ids
        self._state = {}
        group_indices = data.groupby(group_ids).indices
        for column_name in self.feature_name:
            groupped_estimators = {}
            for group_id, idx in group_indices.items():
                source = data.iloc[idx]
                estimator = PolynomialDetrendEstimator()
                estimator.fit(source[column_name], source["_time_idx_"])
                groupped_estimators[group_id] = estimator
            self._state[column_name] = groupped_estimators

    def transform_self(self, data: pd.DataFrame, group_ids):
        # with group ids
        group_indices = data.groupby(group_ids).groups
        for column_name in self.feature_name:
            groupped_estimators = self._state[column_name]
            for group_id, idx in group_indices.items():
                source = data.loc[idx]

                data.loc[idx, column_name] = groupped_estimators[group_id].transform(
                    source[column_name], source["_time_idx_"]
                )

        return data[self.feature_name]

    def inverse_transform_self(self, data: pd.DataFrame, group_ids):
        # with group ids
        group_indices = data.groupby(group_ids).groups
        for column_name in self.feature_name:
            groupped_estimators = self._state[column_name]
            for group_id, idx in group_indices.items():
                source = data.loc[idx]
                data.loc[idx, column_name] = groupped_estimators[group_id].inverse_transform(
                    source[column_name], source["_time_idx_"]
                )
        return data[self.feature_name]
