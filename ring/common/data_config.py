from dataclasses import dataclass, field
from email.errors import CloseBoundaryNotFoundDefect
from typing import List, Dict

from torch import embedding

from .oss_utils import get_bucket_from_oss_url
from .serializer import loads
from .utils import remove_prefix


@dataclass
class Categorical:
    name: List[str] = field(default_factory=list)
    embedding_sizes: List[int] = field(default_factory=list)
    choices: List[List[str]] = field(default_factory=list)


@dataclass
class IndexerConfig:
    name: str
    look_back: int
    look_forward: int


@dataclass
class DataConfig:
    time: str
    freq: str
    targets: List[str]
    indexer: IndexerConfig
    categoricals: Categorical
    group_ids: List[str] = field(default_factory=list)
    static_categoricals: List[str] = field(default_factory=list)
    static_reals: List[str] = field(default_factory=list)
    time_varying_known_categoricals: List[str] = field(default_factory=list)
    time_varying_known_reals: List[str] = field(default_factory=list)
    time_varying_unknown_categoricals: List[str] = field(default_factory=list)
    time_varying_unknown_reals: List[str] = field(default_factory=list)


def dict_to_data_config(cfg: Dict) -> DataConfig:
    indexer = IndexerConfig(
        name=cfg["indexer"]["name"],
        look_back=cfg["indexer"]["look_back"],
        look_forward=cfg["indexer"]["look_forward"],
    )
    cats = Categorical(
        name=cfg["categoricals"]["name"],
        embedding_sizes=cfg["categoricals"]["embedding_sizes"],
        choices=cfg["categoricals"]["choices"],
    )
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
