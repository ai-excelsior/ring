"""
Encoders for encoding categorical variables and scaling continuous data.
"""
from typing import Callable, Dict, List, Tuple, Union, Any
import abc

import numpy as np
import pandas as pd
import torch
import io
import torch.nn.functional as F

from .estimators import Estimator
from .utils import register


NORMALIZERS: Dict[str, "AbstractNormalizer"] = {}


def serialize_normalizer(obj: "AbstractNormalizer"):
    d = obj.serialize()
    d["name"] = obj.__class__.__name__
    return d


def deserialize_normalizer(d: Dict[str, Any]):
    cls: "AbstractNormalizer" = NORMALIZERS[d["name"]]
    return cls.deserialize(d)


def _plus_one(x):
    return x + 1


def _identity(x):
    return x


def _clamp_zero_torch(x):
    return x.clamp(0.0)


def _clamp_zero_np(x):
    return np.clip(x, 0.0, None)


def _logit_np(x):
    return np.log(x / (1.0 - x))


def _expit_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


TRANSFORMATIONS_TORCH = {
    "log": (torch.log, torch.exp),
    "log1p": (torch.log1p, torch.exp),
    "logit": (torch.logit, torch.sigmoid),
    "softplus": (_plus_one, F.softplus),
    "relu": (_identity, _clamp_zero_torch),
}

TRANSFORMATIONS_NUMPY = {
    "log": (np.log, np.exp),
    "log1p": (np.log1p, np.exp),
    "logit": (_logit_np, _expit_np),
    "softplus": (_plus_one, _softplus_np),
    "relu": (_identity, _clamp_zero_np),
}


class AbstractNormalizer(Estimator):
    """
    Basic normalizer that support pre transformation.
    """

    def __init__(
        self,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        self._eps = eps
        self._transformation = transformation

    def preprocess(self, y: Union[pd.Series, torch.Tensor]) -> Union[pd.Series, torch.Tensor]:
        if self._transformation is None:
            return y

        # protect against numerical instabilities
        if isinstance(self._transformation, str) and self._transformation == "logit":
            # need to apply eps slightly differently
            y = y / (1 + 2 * self._eps) + self._eps
        else:
            y = y + self._eps

        TRANSFORMATIONS = TRANSFORMATIONS_TORCH if isinstance(y, torch.Tensor) else TRANSFORMATIONS_NUMPY
        return TRANSFORMATIONS.get(self._transformation, self._transformation)[0](y)

    def inverse_postprocess(self, y: Union[pd.Series, torch.Tensor]) -> Union[pd.Series, torch.Tensor]:
        if self._transformation is None:
            return y

        TRANSFORMATIONS = TRANSFORMATIONS_TORCH if isinstance(y, torch.Tensor) else TRANSFORMATIONS_NUMPY
        return TRANSFORMATIONS.get(self._transformation, self._transformation)[1](y)

    @abc.abstractmethod
    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        pass

    @abc.abstractmethod
    def transform_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        pass

    @abc.abstractmethod
    def inverse_transform_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        pass

    @abc.abstractmethod
    def get_norm(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> np.ndarray:
        pass

    def fit(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        data_transformed = self.preprocess(data)
        return self.fit_self(data_transformed, source)

    def fit_transform(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        data_transformed = self.preprocess(data)
        self.fit_self(data_transformed, source)
        return self.transform_self(data_transformed, source, **kwargs)

    def transform(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        data_transformed = self.preprocess(data)
        return self.transform_self(data_transformed, source, **kwargs)

    def inverse_transform(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        data_inversed = self.inverse_transform_self(data, source, **kwargs)
        return self.inverse_postprocess(data_inversed)


@register(NORMALIZERS)
class Normalizer(AbstractNormalizer):
    """
    A basic single series normalizer. this will not normalize values, but supply some basic function for child class like:
        1. get_norm, return norm state by input.
        2. serializer and deserializer helpers.
    """

    def __init__(
        self,
        feature_name=None,
        center=True,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        eps: float = 1e-8,
    ):
        super().__init__(transformation, eps)
        self._center = center
        self.feature_name = feature_name

    def transform_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        assert self._state is not None
        center, scale = self._state

        return (data - center) / scale

    def inverse_transform_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        assert self._state is not None
        center, scale = self._state

        return data * scale + center

    def get_norm(self, data: pd.Series) -> np.ndarray:
        return np.tile(np.asarray(self._state), (len(data), 1))


@register(NORMALIZERS)
class StandardNormalizer(Normalizer):
    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        if self._center:
            self._state = (data.mean(), data.std() + self._eps)
        else:
            self._state = (0.0, data.mean() + self._eps)


@register(NORMALIZERS)
class MinMaxNormalizer(Normalizer):
    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        min = data.min()
        max = data.max()

        if self._center:
            self._state = ((min + max) / 2.0, max - min + self._eps)
        else:
            self._state = (min, max - min + self._eps)


@register(NORMALIZERS)
class QuantileNormalizer(Normalizer):
    def __init__(
        self,
        quantiles=[0.25, 0.75],
        center=True,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        eps: float = 1e-8,
    ):
        super().__init__(center, transformation, eps)
        self._quantiles = quantiles

    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        center = data.median()
        quantiles = [data.quantile(q) for q in self._quantiles]
        scale = (max(quantiles) - min(quantiles)) / float(len(quantiles))

        if self._center:
            self._state = (center, scale + self._eps)
        else:
            self._state = (0.0, center + self._eps)


@register(NORMALIZERS)
class GroupNormalizer(AbstractNormalizer):
    """
    Normalizer in each group by group_ids
    """

    def __init__(
        self,
        group_ids: List[str] = [],
        feature_name: str = None,
        center=True,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        eps: float = 1e-8,
    ):
        super().__init__(transformation, eps)
        assert len(group_ids) > 0, "group_ids is required for GroupNormalizer"

        self._group_ids = group_ids
        self._center = center
        self._feature_name = feature_name

    def transform_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        assert self._state is not None
        state = source[self._group_ids].join(self._state, on=self._group_ids)
        return (data - state["center"]) / state["scale"]

    def inverse_transform_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs) -> pd.Series:
        assert self._state is not None
        state = source[self._group_ids].join(self._state, on=self._group_ids)
        return data * state["scale"] + state["center"]

    def get_norm(self, source: pd.DataFrame) -> np.ndarray:
        return source[self._group_ids].join(self._state, on=self._group_ids)[["center", "scale"]].to_numpy()

    def serialize(self):
        return {
            "params": self.get_params(),
            "state": self._state.to_csv(),
        }

    @classmethod
    def deserialize(cls, config: Dict):
        params = config.get("params", {})
        state = config.get("state", None)
        group_ids = params.get("group_ids", [])

        state = None if state is None else pd.read_csv(io.StringIO(state), index_col=group_ids)
        self = cls(**params)
        self._state = state

        return self


@register(NORMALIZERS)
class GroupStardardNormalizer(GroupNormalizer):
    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        df_center_scale = (
            source[self._group_ids]
            .assign(y=data)
            .groupby(self._group_ids, observed=True)
            .agg(center=("y", "mean"), scale=("y", "std"))
        )
        if self._center:
            df_center_scale["scale"] += self._eps
            self._state = df_center_scale
        else:
            df_center_scale["scale"] = df_center_scale["center"] + self._eps
            df_center_scale["center"] = 0.0
            self._state = df_center_scale


@register(NORMALIZERS)
class GroupMinMaxNormalizer(GroupNormalizer):
    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        df_min_max = (
            source[self._group_ids]
            .assign(y=data)
            .groupby(self._group_ids, observed=True)
            .agg(min=("y", "min"), max=("y", "max"))
        )

        if self._center:
            df_min_max["center"] = (df_min_max["min"] + df_min_max["max"]) / 2.0
            df_min_max["scale"] = df_min_max["max"] - df_min_max["min"] + self._eps
            df_min_max.drop(columns=["min", "max"], inplace=True)
            self._state = df_min_max
        else:
            df_min_max["center"] = df_min_max["min"]
            df_min_max["scale"] = df_min_max["max"] - df_min_max["min"] + self._eps
            df_min_max.drop(columns=["min", "max"], inplace=True)
            self._state = df_min_max


@register(NORMALIZERS)
class GroupQuantileNormalizer(GroupNormalizer):
    def __init__(
        self,
        group_ids: List[str] = [],
        quantiles=[0.25, 0.75],
        center=True,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        eps: float = 1e-8,
    ):
        super().__init__(group_ids, center, transformation, eps)
        self._quantiles = quantiles

    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        def scale_fn(x: pd.Series):
            quantiles = [x.quantile(q) for q in self._quantiles]
            return max(quantiles) - min(quantiles) / float(len(quantiles))

        df = (
            source[self._group_ids]
            .assign(y=data)
            .groupby(self._group_ids, observed=True)
            .agg(center=("y", "median"), scale=("y", scale_fn))
        )

        if self._center:
            df["scale"] += self._eps
            self._state = df
        else:
            df["scale"] = df["center"] + self._eps
            df["center"] = 0.0
            self._state = df
