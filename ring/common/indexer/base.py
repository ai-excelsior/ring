import abc
import pandas as pd
import inspect
from typing import Dict, Any
from copy import deepcopy


class BaseIndexer:
    """
    索引类，用来创建时序数据的索引。
    当使用滑动窗口的时候，每个索引即为一个窗口。
    索引类必须实现如下接口：
        1. __len__ 长度，例如：总共多少个窗口。
        2. __getitem__ 获取每个元素，一个元素即一个时序片段，一个片段为原始数据中的起始位置[start, breakpoint, end)。例如：总共多少个窗口。
        3. index(data) 根据数据重新创建索引
    """

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        return 0

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abc.abstractmethod
    def index(self, data: pd.DataFrame, evaluate_mode: bool = False, **kwargs):
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.
        """
        return {
            name: getattr(self, f"_{name}")
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["self"]
        }

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any], **kwargs):
        parameters = deepcopy(parameters)
        parameters.update(**kwargs)
        return cls(**parameters)
