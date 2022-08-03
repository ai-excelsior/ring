from typing import Dict, Any, List, Union

from .estimators.base import Estimator, AbstractDetrendEstimator, PolynomialDetrendEstimator
from .autoperiod import Autoperiod
from .utils import register
import copy
import numpy as np
import pandas as pd

SEASONALITY: Dict[str, "AbsrtactDetrend"] = {}


def create_detrender_from_cfg(detrend: bool, group_ids: List[str], targets: List[str]):
    return (
        GroupDetrendTargets(feature_name=targets)
        if detrend and group_ids
        else DetrendTargets(feature_name=targets)
        if detrend
        else AbsrtactDetrend(feature_name=targets)
    )


def create_lags_from_cfg(lags: Union[Dict, None], group_ids: List[str], targets: List[str]):
    if isinstance(lags, Dict):  # need lags
        lags.update(
            **{k: AbstractDetectTargetLags(feature_name=k, lags=lags[k]) for k in targets if k in lags}
        )
        lags.update(
            **{k: DetectTargetLags(feature_name=k) for k in targets if (k not in lags) and (not group_ids)}
        )
        lags.update(
            **{k: GroupDetectTargetLags(feature_name=k) for k in targets if (k not in lags) and group_ids}
        )
        return lags
    elif not lags:  # dont need lags
        return {}
    else:
        raise TypeError(f"data_cfg.lags can only be dict or None, but get {type(lags)}")


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
        return self.fit_self(data, group_ids)

    def transform(self, data: pd.DataFrame, group_ids, **kwargs):
        return self.transform_self(data, group_ids, **kwargs)

    def inverse_transform(self, data: pd.DataFrame, group_ids, **kwargs):
        return self.inverse_transform_self(data, group_ids, **kwargs)

    def fit_self(self, data: pd.DataFrame, group_ids):
        self._state = {
            target_column_name: AbstractDetrendEstimator() for target_column_name in self.feature_name
        }

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
        if self.fitted:
            return
        self._state = {
            target_column_name: PolynomialDetrendEstimator() for target_column_name in self.feature_name
        }
        for column_name, estimator in self._state.items():
            estimator.fit(data[column_name], data["_time_idx_"])

    def transform_self(self, data: pd.DataFrame, group_ids):
        assert self._state is not None
        for column_name, estimator in self._state.items():
            data[column_name] = estimator.transform(data[column_name], data["_time_idx_"])
        return data[self.feature_name]

    def inverse_transform_self(self, data: pd.DataFrame, group_ids):
        assert self._state is not None
        for column_name, estimator in self._state.items():
            data[column_name] = estimator.inverse_transform(data[column_name], data["_time_idx_"])

        return data[self.feature_name]


@register(SEASONALITY)
class GroupDetrendTargets(DetrendTargets):
    def fit_self(self, data: pd.DataFrame, group_ids):
        if self.fitted:
            return
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


@register(SEASONALITY)
class AbstractDetectTargetLags(Estimator):
    def __init__(self, feature_name=None, lags=None) -> None:
        super().__init__()
        self.feature_name = feature_name
        self.lags = lags

    def fit_self(self, data: pd.DataFrame, group_ids: List = []):
        if self.fitted:
            return
        self._detect_lags(data, group_ids)
        self._state = {f"{self.feature_name}_lagged_by_{v}": v for v in self.lags}

    def _detect_lags(self, data: pd.DataFrame, group_ids: List = []):
        pass

    def _is_unique(self, s: pd.Series):
        a = s.to_numpy()
        return (a[0] == a).all()

    def add_lags(self, data: pd.DataFrame, group_ids: List = []):
        # find lags
        self.fit_self(data, group_ids)
        # return pd.dataframe
        for _, v in self._state.items():
            pass


class DetectTargetLags(AbstractDetectTargetLags):
    def _detect_lags(self, data: pd.DataFrame, group_ids: List = []):
        """Detect data lags"""
        lags = {}
        p = Autoperiod(data["_time_idx_"].values, data[data.columns[1]].values)
        if p.period is not None:
            lags[data.columns[1]] = p.period
        return lags

    def _merge_lags(self, new_lags: Dict[str, float]):
        """Merge lags with meta lags"""
        lags = copy.deepcopy(self.meta.get("lags", {}))
        for column_name, lag_value in new_lags.items():
            lag_values = lag_value if isinstance(lag_value, list) else [lag_value]
            if column_name in lags:
                for v in lag_values:
                    # dict
                    if isinstance(lags[column_name], dict) and v not in lags[column_name].values():
                        lags[column_name][f"_{column_name}_lagged_by_{v}_"] = v
                    # list
                    if isinstance(lags[column_name], list) and v not in lags[column_name]:
                        lags[column_name].append(v)
            else:
                lags[column_name] = lag_values
        return lags


class GroupDetectTargetLags(DetectTargetLags):
    def _detect_lags(self, data: pd.DataFrame):
        """Detect data lags"""
        lags = {}

        lags_df = data.groupby(group_ids).aggregate(
            {
                column_name: lambda x: Autoperiod(
                    np.arange(len(x)), x.values, **self._config.get(column_name, {})
                ).period
                for column_name in targets
            }
        )

        lags_df.fillna(0)
        for column_name in targets:
            if self._is_unique(lags_df[column_name]):
                lags[column_name] = lags_df[column_name].iloc[0]
            else:
                lags[column_name] = np.unique(lags_df[column_name]).tolist()

        # filter zero values
        for column_name, lag_value in lags.items():
            if isinstance(lag_value, list):
                lags[column_name] = [round(lag) for lag in lag_value if lag != 0 and not np.isnan(lag)]
            elif lag_value == 0:
                del lags[column_name]

        return lags

    def _merge_lags(self, new_lags: Dict[str, float]):
        """Merge lags with meta lags"""
        lags = copy.deepcopy(self.meta.get("lags", {}))
        for column_name, lag_value in new_lags.items():
            lag_values = lag_value if isinstance(lag_value, list) else [lag_value]
            if column_name in lags:
                for v in lag_values:
                    # dict
                    if isinstance(lags[column_name], dict) and v not in lags[column_name].values():
                        lags[column_name][f"_{column_name}_lagged_by_{v}_"] = v
                    # list
                    if isinstance(lags[column_name], list) and v not in lags[column_name]:
                        lags[column_name].append(v)
            else:
                lags[column_name] = lag_values
        return lags
