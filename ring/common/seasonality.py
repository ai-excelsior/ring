from typing import Dict, Any, List, Union

from .estimators.base import Estimator, AbstractDetrendEstimator, PolynomialDetrendEstimator
from .autoperiod import Autoperiod
from .utils import register
import numpy as np
import pandas as pd

SEASONALITY: Dict[str, "AbsrtactDetrend"] = {}
TIME_IDX = "_time_idx_"


def create_detrender_from_cfg(no_lags: bool, detrend: bool, group_ids: List[str], targets: List[str]):
    # if `data_cfg.lags` is not None, detrender will not be `AbsrtactDetrend`
    return (
        GroupDetrendTargets(feature_name=targets)
        if (detrend or not no_lags) and group_ids
        else DetrendTargets(feature_name=targets)
        if detrend or not no_lags
        else AbsrtactDetrend(feature_name=targets)
    )


def create_lags_from_cfg(lags: Union[Dict, None], group_ids: List[str], targets: List[str]):
    lags_out = {}
    if isinstance(lags, Dict):  # need lags
        lags_out.update(
            **{k: AbstractDetectTargetLags(feature_name=k, lags=lags[k]) for k in targets if k in lags}
        )
        lags_out.update(
            **{k: DetectTargetLags(feature_name=k) for k in targets if (k not in lags) and (not group_ids)}
        )
        lags_out.update(
            **{k: GroupDetectTargetLags(feature_name=k) for k in targets if (k not in lags) and group_ids}
        )
        return lags_out
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


def serialize_lags(obj: List["AbstractDetectTargetLags"]):

    d = obj.serialize()
    d["name"] = obj.__class__.__name__
    return d


def deserialize_lags(d: Dict[str, Any]):
    cls: List["AbstractDetectTargetLags"] = [SEASONALITY[i["name"]] for i in d]
    out = {}
    for i, lag in enumerate(d):
        out.update({lag["params"].get("feature_name"): cls[i].deserialize(lag)})
    return out


@register(SEASONALITY)
class AbsrtactDetrend(Estimator):
    """do nothing to detrend"""

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
    """do detrend, no groups"""

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
            estimator.fit(data[column_name], data[TIME_IDX])

    def transform_self(self, data: pd.DataFrame, group_ids):
        assert self._state is not None
        for column_name, estimator in self._state.items():
            data[column_name] = estimator.transform(data[column_name], data[TIME_IDX])
        return data[self.feature_name]

    def inverse_transform_self(self, data: pd.DataFrame, group_ids):
        assert self._state is not None
        for column_name, estimator in self._state.items():
            data[column_name] = estimator.inverse_transform(data[column_name], data[TIME_IDX])

        return data[self.feature_name]


@register(SEASONALITY)
class GroupDetrendTargets(DetrendTargets):
    """do detrend, has groups"""

    def __init__(self, feature_name=None) -> None:
        super().__init__(feature_name=feature_name)

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
                estimator.fit(source[column_name], source[TIME_IDX])
                groupped_estimators[group_id] = estimator
            self._state[column_name] = groupped_estimators

    def _transform_seperately(self, data: pd.DataFrame, column_name):
        estimators = self._state[column_name][data.name]
        data[column_name] = estimators.transform(data[column_name], data[TIME_IDX])
        return data[self.feature_name]

    def transform_self(self, data: pd.DataFrame, group_ids):
        for column_name in self.feature_name:
            data = data.groupby(group_ids).apply(self._transform_seperately, column_name)
        return data[self.feature_name]

    def _inverse_transform_seperately(self, data: pd.DataFrame, column_name):
        estimators = self._state[column_name][data.name]
        data[column_name] = estimators.inverse_transform(data[column_name], data[TIME_IDX])
        return data[self.feature_name]

    def inverse_transform_self(self, data: pd.DataFrame, group_ids):
        for column_name in self.feature_name:
            data = data.groupby(group_ids).apply(self._inverse_transform_seperately, column_name)
        return data[self.feature_name]


@register(SEASONALITY)
class AbstractDetectTargetLags(Estimator):
    """do nothing to detect lags, use input config as default lags"""

    def __init__(self, feature_name=None, lags=[]) -> None:
        super().__init__()
        self.feature_name = feature_name
        self.lags = [i for i in lags if i > 0 and isinstance(i, int)]

    def fit_self(self, data: pd.DataFrame, group_ids: List = []):
        if self.fitted:
            return
        self._detect_lags(data, group_ids)
        self._state = {f"{self.feature_name}_lagged_by_{v}": v for v in self.lags}

    def _detect_lags(self, data: pd.DataFrame, group_ids: List = []):
        pass

    def add_lags(self, data: pd.DataFrame, group_ids: List = []):
        # find lags
        self.fit_self(data, group_ids)
        # return pd.dataframe
        if self._state:
            if group_ids:
                tmp_data = pd.DataFrame()
                for _, source in data.groupby(group_ids):
                    for k, v in self._state.items():
                        source[k] = source[self.feature_name].shift(v)
                    tmp_data = pd.concat([tmp_data, source])
                return tmp_data[[k for k, _ in self._state.items()]]
            else:
                for k, v in self._state.items():
                    data[k] = data[self.feature_name].shift(v)
            return data[[k for k, _ in self._state.items()]]
        else:
            return pd.DataFrame()


@register(SEASONALITY)
class DetectTargetLags(AbstractDetectTargetLags):
    """detect lags, no groups"""

    def _detect_lags(self, data: pd.DataFrame, group_ids: List = []):
        """Detect data lags"""
        p = Autoperiod(data[TIME_IDX].values, data[data.columns[1]].values)
        if p.period is not None:
            self.lags += p.period_list


@register(SEASONALITY)
class GroupDetectTargetLags(AbstractDetectTargetLags):
    """detect lags, has groups"""

    def _is_unique(self, s: pd.Series):
        return list(np.unique(np.array(list(filter(lambda x: x, s)))))

    def _detect_lags(self, data: pd.DataFrame, group_ids: List = []):
        """Detect data lags"""
        lags_df = data.groupby(group_ids).aggregate(
            {data.columns[1]: lambda x: Autoperiod(np.arange(len(x)), x.values).period_list}
        )

        self.lags += self._is_unique(lags_df[data.columns[1]])
