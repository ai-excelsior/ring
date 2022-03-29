from torch import nn
from .dataset import TimeSeriesDataset


class BaseModel(nn.Module):
    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "BaseModel":
        raise NotImplementedError()
