from typing import Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
import statsmodels.api as sm
import pandas as pd
import numpy as np
from ..serializer import pickle_dumps, pick_loads
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.filters.hp_filter import hpfilter


def log_func(x, base=np.e):
    return np.log(x + 1) / np.log(base)


class Estimator(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self._state = None

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, getattr(self, f"_{key}", None))
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    @property
    def fitted(self):
        return self._state is not None

    def _assert_fitted(self):
        assert self.fitted, "Can not transform before fitted"

    def fit(self):
        pass

    def transform(self):
        self._assert_fitted()

    def fit_transform(self, *args, **kwargs):
        if not self.fitted:
            self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def inverse_transform(self, *args, **kwargs):
        self._assert_fitted()

        return args[0] if len(args) > 0 else None

    def serialize(self):
        return {
            "params": self.get_params(),
            "state": self._state,
        }

    @classmethod
    def deserialize(cls, config: Dict):
        params = config.get("params", {})
        state = config.get("state", None)

        this = cls(**params)
        this._state = state

        return this


class BypassEstimator(BaseEstimator):
    """Do nothing estimator"""

    def __init__(self, *_, **__) -> None:
        super().__init__()

    @property
    def fitted(self):
        return True

    def fit(self, *_, **__):
        pass

    def transform(self, data, *args):
        if len(args) > 0:
            return tuple(data, *args)
        else:
            return data

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def serialize(self):
        return dict()

    @classmethod
    def deserialize(cls, *args, **kwargs):
        return cls()

    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """


class AbstractDetrendEstimator(Estimator):
    """
    A PolynomialDetrendEstimator that try to find best degree to fit the passed-in data and remove the trend.
    """

    def __init__(self, max_degress=4, **kwargs) -> None:
        super().__init__()
        self._max_degress = max_degress

    def transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        self._assert_fitted()
        return data

    def inverse_transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        self._assert_fitted()
        return data


class LogDetrendEstimator(AbstractDetrendEstimator):
    """make sure data is all positive

    Args:
        LogDetrendEstimator (_type_): try to fit a log trend
    """

    def __init__(self, max_degress=[2, np.e, 10], **kwargs) -> None:
        super().__init__()
        self._max_degress = max_degress
        self._degree = None

    def fit(self, data: pd.Series, index: pd.Series):
        df = pd.DataFrame({"x": index, "y": data})
        df = df[~df.y.isnull()]  # filter out missing values
        x, y = df.x.values.reshape(-1, 1), df.y.values.reshape(-1, 1)
        best_score = np.inf
        best_model = None
        best_base = None

        for base in self._max_degress:
            log_feature_model = FunctionTransformer(log_func, kw_args={"base": base})
            feature_matrix = log_feature_model.fit_transform(x)
            ols_model = sm.OLS(y, feature_matrix).fit()
            scores = np.sum((feature_matrix - y) ** 2) ** 0.5
            if scores < best_score:
                best_model = (log_feature_model, ols_model)
                best_score = scores
                best_base = base

        # at least one coefficient of the linear is nonzero with statistical significance
        if best_model[1].f_pvalue <= 0.01:
            self._state = best_model
            self._degree = best_base
        else:
            self._state = None
            self._degree = None

    def get_trend(self, index: pd.Series) -> np.ndarray:
        self._assert_fitted()
        log_feature_model, ols_model = self._state
        return ols_model.predict(log_feature_model.transform(index.to_numpy()[:, np.newaxis]))

    def transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        return data - self.get_trend(index) if self._state is not None else data

    def inverse_transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        return data + self.get_trend(index) if self._state is not None else data

    def get_degree(self):
        return self._degree

    def serialize(self):
        return {
            "params": self.get_params(),
            "state": (pickle_dumps(self._state[0]), pickle_dumps(self._state[1]))
            if self._state is not None
            else None,
        }

    @classmethod
    def deserialize(cls, config: Dict):
        params = config.get("params", {})
        state = config.get("state", None)
        if state is not None:
            state = (pick_loads(state[0]), pick_loads(state[1]))
        this = cls(**params)
        this._state = state

        return this


class PolynomialDetrendEstimator(AbstractDetrendEstimator):
    """
    A PolynomialDetrendEstimator that try to find best degre e to fit the passed-in data and remove the trend.
    """

    def __init__(self, max_degress=4, **kwargs) -> None:
        super().__init__()
        self._max_degress = max_degress
        self._degree = None

    def fit(self, data: pd.Series, index: pd.Series):
        df = pd.DataFrame({"x": index, "y": data})
        df = df[~df.y.isnull()]  # filter out missing values
        x, y = df.x.values.reshape(-1, 1), df.y.values.reshape(-1, 1)
        best_score = np.inf
        best_model = None
        best_degree = None

        for degree in range(1, self._max_degress + 1):
            polynomial_feature_model = PolynomialFeatures(degree=degree)
            feature_matrix = polynomial_feature_model.fit_transform(x)
            ols_model = sm.OLS(y, feature_matrix).fit()
            if ols_model.aic < best_score:
                best_model = (polynomial_feature_model, ols_model)
                best_score = ols_model.aic
                best_degree = degree

        # at least one coefficient of the linear is nonzero with statistical significance
        if best_model[1].f_pvalue <= 0.01:
            self._state = best_model
            self._degree = best_degree
        else:
            self._state = None
            self._degree = None

    def get_trend(self, index: pd.Series) -> np.ndarray:
        self._assert_fitted()
        polynomial_feature_model, ols_model = self._state
        return ols_model.predict(polynomial_feature_model.transform(index.to_numpy()[:, np.newaxis]))

    def transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        return data - self.get_trend(index) if self._state is not None else data

    def inverse_transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        return data + self.get_trend(index) if self._state is not None else data

    def get_degree(self):
        return self._degree

    def serialize(self):
        return {
            "params": self.get_params(),
            "state": (pickle_dumps(self._state[0]), pickle_dumps(self._state[1]))
            if self._state is not None
            else None,
        }

    @classmethod
    def deserialize(cls, config: Dict):
        params = config.get("params", {})
        state = config.get("state", None)
        if state is not None:
            state = (pick_loads(state[0]), pick_loads(state[1]))
        this = cls(**params)
        this._state = state

        return this


class HPfilterDetrendEstimator(AbstractDetrendEstimator):
    """

    Args:
        Hodrick-Prescott filter DetrendEstimator (_type_)
    """

    def __init__(self, freq=None) -> None:
        super().__init__()
        self._max_degress = freq

    def _detect_freq(self, freq):
        freq = to_offset(freq).name
        if "AS-" in freq or "A-" in freq or freq == "AS" or freq == "A":
            return 6.25
        elif "QS-" in freq or "Q-" in freq or freq == "QS" or freq == "Q":
            return 1600
        elif "MS-" in freq or "M-" in freq or freq == "MS" or freq == "M":
            return 129600
        else:
            return 10e6

    def fit(self, data: pd.Series, index: pd.Series):
        self._state = self._detect_freq(self._max_degress)

    def get_trend(self, x: pd.Series) -> np.ndarray:
        self._assert_fitted()
        lamb = self._state
        return hpfilter(x, lamb)[1]

    def transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        return data - self.get_trend(data) if self._state is not None else data

    def inverse_transform(self, data: pd.Series, index: pd.Series) -> pd.Series:
        return data + self.get_trend(data) if self._state is not None else data

    def get_degree(self):
        return self._state

    def serialize(self):
        return {
            "params": self.get_params(),
            "state": self._state if self._state is not None else None,
        }

    @classmethod
    def deserialize(cls, config: Dict):
        params = config.get("params", {})
        state = config.get("state", None)
        if state is not None:
            state = state
        this = cls(**params)
        this._state = state

        return this
