from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Categorical:
    choices: List[str] = field(default_factory=list)


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
    )
    return data_config
