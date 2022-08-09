from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .oss_utils import get_bucket_from_oss_url
from .serializer import loads
from .utils import remove_prefix


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
    detrend: bool = field(default_factory=bool)
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


def dict_to_data_config(cfg: Dict) -> DataConfig:
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
        time_features=cfg.get("time_features", []),
        detrend=cfg.get("detrend", False),
        lags=cfg.get("lags", None),
    )
    return data_config


def dict_to_data_config_anomal(cfg: Dict) -> DataConfig:
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
        detrend=cfg.get("detrend", {}),
        lags=cfg.get("lags", {}),
    )
    return data_config


def url_to_data_config(url: str) -> DataConfig:
    if url.startswith("file://"):
        with open(remove_prefix(url, "file://"), "r") as f:
            return dict_to_data_config(loads(f.read()))
    elif url.startswith("oss://"):
        bucket, key = get_bucket_from_oss_url(url)
        return dict_to_data_config(loads(bucket.get_object(key).read()))

    raise "url should be one of file:// or oss://"


def url_to_data_config_anomal(url: str) -> DataConfig:
    if url.startswith("file://"):
        with open(remove_prefix(url, "file://"), "r") as f:
            return dict_to_data_config_anomal(loads(f.read()))
    elif url.startswith("oss://"):
        bucket, key = get_bucket_from_oss_url(url)
        return dict_to_data_config_anomal(loads(bucket.get_object(key).read()))

    raise "url should be one of file:// or oss://"
