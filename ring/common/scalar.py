from typing import Dict, Any, List

from .estimators import Estimator
import pandas as pd
import abc
import numpy as np
from .utils import register, column_or_1d, _num_samples, _map_to_integer

ScalerS: Dict[str, "AbstractScaler"] = {}


def serialize_scaler(obj: "AbstractScaler"):
    d = obj.serialize()
    d["name"] = obj.__class__.__name__
    return d


def deserialize_scaler(d: Dict[str, Any]):
    cls: "AbstractScaler" = ScalerS[d["name"]]
    return cls.deserialize(d)


class AbstractScaler(Estimator):
    def __init__(self):
        self._state = None

    @abc.abstractmethod
    def fit_self(self, y: pd.Series, embeddings: Dict = {}, **kwargs):
        pass

    @abc.abstractmethod
    def transform_self(self, y: pd.Series, embeddings: Dict = {}, **kwargs) -> pd.Series:
        pass

    @abc.abstractmethod
    def inverse_transform_self(self, y: pd.Series, embeddings: Dict = {}, **kwargs) -> pd.Series:
        pass

    @abc.abstractmethod
    def get_norm(self, y: pd.Series, embeddings: Dict = {}, **kwargs) -> np.ndarray:
        pass

    def fit(self, y: pd.Series, embeddings: Dict = {}, **kwargs):
        return self.fit_self(y, embeddings, **kwargs)

    def fit_transform(self, y: pd.Series, embeddings: Dict = {}, **kwargs):
        self.fit_self(y, **kwargs)
        return self.transform_self(y, **kwargs)

    def transform(self, y: pd.Series, embeddings: Dict = {}, **kwargs):
        return self.transform_self(y, **kwargs)

    def inverse_transform(self, y: pd.Series, embeddings: Dict = {}, **kwargs):
        data_inversed = self.inverse_transform_self(y, **kwargs)
        return data_inversed


@register(ScalerS)
class LabelScaler(AbstractScaler):
    def fit_self(self, y: pd.Series):
        y = column_or_1d(y, warn=True)
        self.classes_ = np.array(["UNKNOWN"] + sorted(set(y)), dtype=y.dtype)
        self._state = "LabelScaler"
        return self

    def transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        assert self._state is not None
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])
        y = _map_to_integer(y, self.classes_)
        return y

    def inverse_transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        assert self._state is not None
        y = column_or_1d(y, warn=True)
        if _num_samples(y) == 0:
            return np.array([])
        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]


@register(ScalerS)
class OrdinalScaler(AbstractScaler):
    def fit_self(self, y: pd.Series, reverse=False):
        y = column_or_1d(y, warn=True)
        # in appear order
        self.classes_ = np.array(["UNKNOWN"] + sorted(list(set(y)), key=list(y).index, reverse=reverse))
        self._state = "OrdinalScaler"
        return self

    def transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        assert self._state is not None
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])
        y = _map_to_integer(y, self.classes_)
        return y

    def inverse_transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        assert self._state is not None
        y = column_or_1d(y, warn=True)
        if _num_samples(y) == 0:
            return np.array([])
        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]
