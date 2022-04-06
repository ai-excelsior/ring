import torch
import pandas as pd
import os
import numpy as np

from typing import Any, Dict, List, Union
from pandas.tseries.frequencies import to_offset


def to_prediction(y_pred: torch.Tensor, quantiles: List[float] = None):
    """
    Convert an either quantiled prediction to a point of prediction
        With Tensor(batch_size, num_sequences, quantiles) -> Tensor(batch_size, num_sequences)
        With Tensor(batch_size, num_sequences, 1) -> Tensor(batch_size, num_sequences)
        With Tensor(batch_size, num_sequences) -> Tensor(batch_size, num_sequences)
    """
    if y_pred.ndim == 3:
        if quantiles is None:
            assert y_pred.size(-1) == 1, "Prediction should only have one extra dimension"
            return y_pred[..., 0]
        else:
            return y_pred.mean(-1)
    return y_pred


def to_quantiles(y_pred: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
    """
    Convert a point of prediction into a quantile prediction.
        With Tensor(batch_size, num_sequences) -> Tensor(batch_size, num_sequences, 1)  # add a quantile dim
        With Tensor(batch_size, num_sequences, 1) -> Tensor(batch_size, num_sequences, 1)  # do nothing
        With Tensor(batch_size, num_sequences, num_quantiles) -> Tensor(batch_size, num_sequences, num_quantiles) # with new quantiled prediction
    """
    if y_pred.ndim == 2:
        return y_pred.unsqueeze(-1)

    if y_pred.ndim == 3:
        if y_pred.size(2) > 1:  # single dimension means all quantiles are the same
            assert quantiles is not None, "quantiles are not defined"

            return torch.quantile(y_pred, torch.tensor(quantiles, device=y_pred.device), dim=2).permute(
                1, 2, 0
            )
        return y_pred

    raise ValueError(f"prediction has 1 or more than 3 dimensions: {y_pred.ndim}")


def extract_n_off_targets(x: torch.Tensor, target_indexes: torch.Tensor, lengths: torch.Tensor = None, n=1):
    """
    extract n off targets from x, default n = 1.
    x: Tensor(batch_size, sequence_length, num_features)
    lengths: Tensor(batch_size)
    target_indexes: Tensor(num_targets)
    """
    batch_size = x.size(0)
    sequence_length = x.size(1)

    if lengths is None:
        lengths = torch.tile(torch.tensor(sequence_length), (batch_size,)).to(x.device)

    return x[torch.arange(batch_size, device=x.device), lengths - n, target_indexes]


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai). Magic!!!

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n ** 0.56), max_size)
    else:
        return 1


def between_time(
    data: pd.DataFrame,
    time_column: str,
    start: Union[pd.Timestamp, str],
    end: Union[pd.Timestamp, str],
    include_start=True,
    include_end=True,
):
    index = data.index if time_column == "index" else data[time_column]

    if include_start and include_end:
        return data.iloc[(index <= end) & (index >= start)]
    elif include_start and not include_end:
        return data.iloc[(index < end) & (index >= start)]
    elif not include_start and include_end:
        return data.iloc[(index <= end) & (index > start)]
    else:
        return data.iloc[(index < end) & (index > start)]


def detect_pandas_categorical_columns(data: pd.DataFrame):
    columns = data.columns
    types = data.dtypes
    return [
        column_name
        for column_name in columns
        if types[column_name].kind != "f" and types[column_name].kind != "i"
    ]


def reduce_multi_label_metrics(metrics: Dict, monitor: str, labels: List[str]) -> torch.Tensor:
    metric_values = [metrics.get(f"{label} {monitor}") for label in labels]
    metric_values = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in metric_values]
    metric_tensor = torch.stack(metric_values)

    return torch.mean(metric_tensor)


def register(plugin: Dict[str, Any]):
    def wrapper_register(cls):
        plugin[cls.__name__] = cls
        return cls

    return wrapper_register


def get_latest_updated_file(files: List[str]):
    if len(files) == 0:
        return None

    times = [os.path.getmtime(filepath) for filepath in files]
    max_idx = np.argmax(times)
    return files[max_idx]


def get_default_embedding_size(n: int):
    if n > 2:
        return round(1.6 * n ** 0.56)
    else:
        return 1


def diff_year(d1, d2, n=1):
    return (d1.year - d2.year) // n


def diff_quarter(d1, d2, n=1):
    return ((d1.year - d2.year) * 12 + d1.month - d2.month) // (n * 3)


def diff_month(d1, d2, n=1):
    return ((d1.year - d2.year) * 12 + d1.month - d2.month) // n


def diff_week(d1, d2, n=1):
    return (d1 - d2).days // (n * 7)


def diff_hour(d1, d2, n=1):
    return (d1 - d2).seconds // (n * 3600)


def diff_miniute(d1, d2, n=1):
    return (d1 - d2).seconds // (n * 60)


def add_time_idx(data: pd.DataFrame, time_column_name: str, freq: str = None, time_idx_name="_time_idx_"):
    if time_column_name is not None:
        time_column = pd.to_datetime(data[time_column_name]).dt.tz_localize(None)
        freq = freq or pd.infer_freq(time_column)
    else:
        time_column = pd.to_datetime(data.index.to_series()).dt.tz_localize(None)
        freq = freq or pd.infer_freq(time_column)

    assert freq is not None, "Auto frequency infer failed, please provide it manully."

    offset = to_offset(freq)
    start_time = time_column.min()

    # TODO: more offset should be supported in here.
    if "AS-" in offset.name or "A-" in offset.name or offset.name == "AS" or offset.name == "A":
        time_idx = time_column.apply(lambda x: diff_year(x, start_time, offset.n))
    elif "QS-" in offset.name or "Q-" in offset.name or offset.name == "QS" or offset.name == "Q":
        time_idx = time_column.apply(lambda x: diff_quarter(x, start_time, offset.n))
    elif "MS-" in offset.name or "M-" in offset.name or offset.name == "MS" or offset.name == "M":
        time_idx = time_column.apply(lambda x: diff_month(x, start_time, offset.n))
    elif "W-" in offset.name:
        time_idx = time_column.apply(lambda x: diff_week(x, start_time, offset.n))
    elif "-" not in offset.name and offset.name[-1] == "H":
        time_idx = time_column.apply(lambda x: diff_hour(x, start_time, offset.n))
    elif "-" not in offset.name and offset.name[-1] == "T":
        time_idx = time_column.apply(lambda x: diff_miniute(x, start_time, offset.n))
    else:  # seconds, microseconds, nanosecongs, days, businiess-days, no-time column
        time_idx = ((time_column - start_time) / offset).astype(int)

    return data.assign(**{time_idx_name: time_idx})


def remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]
