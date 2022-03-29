from typing import Dict
from sklearn.base import BaseEstimator, TransformerMixin


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
