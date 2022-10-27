from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pandas as pd
from .oss_utils import get_bucket_from_oss_url
from .serializer import loads
from .utils import remove_prefix
from .data_utils import read_from_url


@dataclass
class Categorical:
    name: str = field(default_factory=list)
    embedding_size: Optional[int] = field(default_factory=int)
    choices: List[str] = field(default_factory=list)


@dataclass
class IndexerConfig:
    name: str
    look_back: int
    look_forward: int


@dataclass
class AnomalIndexerConfig:
    name: str
    steps: int


@dataclass
class DataConfig:
    time: str
    freq: str
    targets: List[str]
    indexer: IndexerConfig
    categoricals: List[Categorical] = field(default_factory=list)
    group_ids: List[str] = field(default_factory=list)
    static_categoricals: List[str] = field(default_factory=list)
    static_reals: List[str] = field(default_factory=list)
    time_varying_known_categoricals: List[str] = field(default_factory=list)
    time_varying_known_reals: List[str] = field(default_factory=list)
    time_varying_unknown_categoricals: List[str] = field(default_factory=list)
    time_varying_unknown_reals: List[str] = field(default_factory=list)
    time_features: List[str] = field(default_factory=list)
    detrend: str = field(default_factory=str)
    lags: Dict = field(default_factory=dict)


@dataclass
class AnomalDataConfig:
    time: str
    freq: str
    indexer: IndexerConfig
    categoricals: List = field(default_factory=list)
    lags: Dict = field(default_factory=dict)
    group_ids: List[str] = field(default_factory=list)
    static_categoricals: List[str] = field(default_factory=list)
    cont_features: List[str] = field(default_factory=list)
    cat_features: List[str] = field(default_factory=list)
    time_features: List[str] = field(default_factory=list)


def dict_to_data_cfg(cfg: Dict) -> DataConfig:
    indexer = IndexerConfig(
        name=cfg["indexer"]["name"],
        look_back=cfg["indexer"]["look_back"],
        look_forward=cfg["indexer"]["look_forward"],
    )
    cats = [Categorical(**item) for item in cfg["categoricals"]]
    data_config = DataConfig(
        time=cfg["time"],
        freq=cfg["freq"],
        targets=cfg["targets"],
        indexer=indexer,
        group_ids=cfg.get("group_ids", []),
        static_categoricals=cfg.get("static_categoricals", []),
        static_reals=cfg.get("static_reals", []),
        time_varying_known_categoricals=cfg.get("time_varying_known_categoricals", []),
        time_varying_known_reals=cfg.get("time_varying_known_reals", []),
        time_varying_unknown_categoricals=cfg.get("time_varying_unknown_categoricals", []),
        time_varying_unknown_reals=cfg.get("time_varying_unknown_reals", []),
        categoricals=cats,
        time_features=cfg.get("time_features", None),
        detrend=cfg.get("detrend", False),
        lags=cfg.get("lags", None),
    )
    return data_config


def dict_to_data_cfg_anomal(cfg: Dict) -> DataConfig:
    indexer = AnomalIndexerConfig(
        name=cfg["indexer"]["name"],
        steps=cfg["indexer"]["steps"],
    )

    cats = [Categorical(**item) for item in cfg["categoricals"]]
    data_config = AnomalDataConfig(
        time=cfg["time"],
        freq=cfg["freq"],
        indexer=indexer,
        group_ids=cfg.get("group_ids", []),
        static_categoricals=cfg.get("static_categoricals", []),
        cont_features=cfg.get("cont_features", []),
        cat_features=cfg.get("cat_features", []),
        categoricals=cats,
        time_features=cfg.get("time_features", []),
        detrend=cfg.get("detrend", False),
        lags=cfg.get("lags", {}),
    )
    return data_config


def dict_to_parse(cfg: Dict, *args):
    data_cfg = dict_to_data_cfg(cfg["data_config"])
    data_info = {
        "url": cfg["data_source"]["path"],
        "type": cfg["data_source"]["type"],
        "parse_dates": [] if data_cfg.time is None else [data_cfg.time],
        "time_range": args,
        "dtype": data_cfg.group_ids,
        "time": data_cfg.time,
    }
    return data_cfg, data_info


def dict_to_parse_anomal(cfg: Dict, *args):
    data_cfg = dict_to_data_cfg_anomal(cfg["data_config"])
    data_info = {
        "url": cfg["data_source"]["path"],
        "type": cfg["data_source"]["type"],
        "parse_dates": [] if data_cfg.time is None else [data_cfg.time],
        "time_range": args,
        "dtype": data_cfg.group_ids,
        "time": data_cfg.time,
    }
    return data_cfg, data_info


def info_to_data(cfg: Dict):
    time = cfg.pop("time")
    time_range = cfg.pop("time_range")
    data = read_from_url(**cfg) if cfg.pop("type") == "file" else None
    time_range = [
        data[time].max()
        if (not obj and i % 2 == 1)
        else data[time].min()
        if (not obj and i % 2 == 0)
        else pd.to_datetime(obj)
        for i, obj in enumerate(time_range)
    ]
    if len(time_range) == 2:
        data = data[(data[time] >= time_range[0]) & (data[time] <= time_range[1])]
        data.sort_values([*cfg["dtype"], time], ignore_index=True)
    elif len(time_range) == 4:
        data_train = data[(data[time] >= time_range[0]) & (data[time] <= time_range[1])]
        data_train.sort_values([*cfg["dtype"], time], ignore_index=True, inplace=True)
        data_valid = data[(data[time] >= time_range[2]) & (data[time] <= time_range[3])]
        data_valid.sort_values([*cfg["dtype"], time], ignore_index=True, inplace=True)
        data = (data_train, data_valid)
    else:
        raise ValueError("start_time dont match with end_time")
    return data


def url_to_data_config(url: str, *args) -> DataConfig:
    if url.startswith("file://"):
        with open(remove_prefix(url, "file://"), "r") as f:
            cfg_info, data_info = dict_to_parse(loads(f.read()), *args)
            return cfg_info, info_to_data(data_info)
    elif url.startswith("oss://"):
        bucket, key = get_bucket_from_oss_url(url)
        cfg_info, data_info = dict_to_parse(loads(bucket.get_object(key).read()), *args)
        return cfg_info, info_to_data(data_info)

    raise "url should be one of file:// or oss://"


def url_to_data_config_anomal(url: str, *args) -> DataConfig:
    if url.startswith("file://"):
        with open(remove_prefix(url, "file://"), "r") as f:
            cfg_info, data_info = dict_to_parse_anomal(loads(f.read()), *args)
            return cfg_info, info_to_data(data_info)
    elif url.startswith("oss://"):
        bucket, key = get_bucket_from_oss_url(url)
        cfg_info, data_info = dict_to_parse_anomal(loads(bucket.get_object(key).read()), *args)
        return cfg_info, info_to_data(data_info)

    raise "url should be one of file:// or oss://"
