from typing import Dict, Any, List, Union
from ..data_config import IndexerConfig, AnomalIndexerConfig
from .base import BaseIndexer
from .slide_window import SlideWindowIndexer, SlideWindowIndexer_bucketSampler, SlideWindowIndexer_fixed

name_to_cls = {
    "slide_window": SlideWindowIndexer,
    "slide_window_fixed": SlideWindowIndexer_fixed,
    "slide_window_bucket": SlideWindowIndexer_bucketSampler,
}


def create_indexer_from_cfg(
    cfg: Union[IndexerConfig, AnomalIndexerConfig], group_ids: List[str] = [], time_idx="_time_idx_"
) -> BaseIndexer:
    if cfg.name == "slide_window":
        return SlideWindowIndexer(
            time_idx=time_idx,
            look_back=cfg.look_back,
            look_forward=cfg.look_forward,
            group_ids=group_ids,
        )
    elif cfg.name == "slide_window_fixed":
        return SlideWindowIndexer_fixed(
            time_idx=time_idx,
            steps=cfg.steps,
            group_ids=group_ids,
        )
    elif cfg.name == "slide_window_bucket":
        return SlideWindowIndexer_bucketSampler(
            time_idx=time_idx,
            look_back=cfg.look_back,
            look_forward=cfg.look_forward,
            group_ids=group_ids,
        )

    raise "Must be one of slide_window ..."


def serialize_indexer(obj: BaseIndexer):
    names = [name for name in name_to_cls if isinstance(obj, name_to_cls[name])]
    if len(names) == 0:
        raise "Must be one of slide_window ..."

    return {"name": names[0], "params": obj.get_parameters()}


def deserialize_indexer(d: Dict[str, Any]):
    name = d["name"]
    cls = name_to_cls.get(name, None)
    params = d["params"]

    assert cls is not None, "Must be one of slide_window ..."

    return cls.from_parameters(params)
