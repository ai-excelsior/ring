from typing import Dict, Any, List

from torch import embedding

from .data_config import Categorical
from .estimators import Estimator
from .utils import get_default_embedding_size
import pandas as pd
import abc
from .utils import register, column_or_1d, _num_samples, _map_to_integer
import numpy as np

ENCODERS: Dict[str, "AbstractEncoder"] = {}


def serialize_encoder(obj: "AbstractEncoder"):
    d = obj.serialize()
    d["name"] = obj.__class__.__name__
    return d


def deserialize_encoder(d: Dict[str, Any]):
    cls: "AbstractEncoder" = ENCODERS[d["name"]]
    return cls.deserialize(d)


def create_encoder_from_cfg(cfg: Categorical) -> Dict:
    # `embedding_sizes` not provided, then it can be deduced
    if len(cfg.embedding_sizes) == 0 and len(cfg.name) != len(cfg.choices):
        raise ValueError("Number of categoricals and list of choices must be the same")
    # if `embedding_sizes` provided, then it should match
    if len(cfg.name) != len(cfg.choices) != len(cfg.embedding_sizes):
        raise ValueError("Number of categoricals, embedding_sizes, list of choices must be the same")
    cat_encoder = {}
    # match the form of parameter `embedding_sizes` in dataset
    for i in range(len(cfg.name)):
        cat_encoder[cfg.name[i]] = (
            len(cfg.choices[i]) + 1,  # add `unknown`
            cfg.embedding_sizes[i]
            if len(cfg.embedding_sizes) != 0
            else get_default_embedding_size(len(cfg.choices[i]) + 1),
        )

    return cat_encoder


class AbstractEncoder(Estimator):
    def __init__(self, feature_name=None):
        super().__init__()
        self.feature_name = feature_name

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
        assert self._state is not None
        return self.transform_self(y, **kwargs)

    def inverse_transform(self, y: pd.Series, embeddings: Dict = {}, **kwargs):
        assert self._state is not None
        data_inversed = self.inverse_transform_self(y, **kwargs)
        return data_inversed


@register(ENCODERS)
class LabelEncoder(AbstractEncoder):
    def fit_self(self, y: pd.Series):
        y = column_or_1d(y, warn=True)
        self._state = ["UNKNOWN"] + sorted(set(y))
        return self

    def transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])
        y = _map_to_integer(y, self._state)
        return y

    def inverse_transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        y = column_or_1d(y, warn=True)
        if _num_samples(y) == 0:
            return np.array([])
        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self._state[y]


@register(ENCODERS)
class OrdinalEncoder(AbstractEncoder):
    def fit_self(self, y: pd.Series, reverse=False):
        y = column_or_1d(y, warn=True)
        # in appear order
        self._state = ["UNKNOWN"] + sorted(list(set(y)), key=list(y).index, reverse=reverse)
        return self

    def transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        assert self._state is not None
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])
        y = _map_to_integer(y, self._state)
        return y

    def inverse_transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        assert self._state is not None
        y = column_or_1d(y, warn=True)
        if _num_samples(y) == 0:
            return np.array([])
        diff = np.setdiff1d(y, np.arange(len(self._state)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self._state[y]
