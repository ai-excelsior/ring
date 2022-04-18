import pandas as pd
from .oss_utils import get_bucket_from_oss_url
from .utils import remove_prefix


def is_parquet(s: str):
    return s.endswith(".parq") or s.endswith(".parquet")


def is_csv(s: str):
    return s.endswith(".csv")


def read_csv(url: str, **config) -> pd.DataFrame:
    if url.startswith("oss://"):
        bucket, key = get_bucket_from_oss_url(url)
        if is_csv(key):
            return pd.read_csv(bucket.get_object(key), thousands=",", **config)
        elif is_parquet(key):
            return pd.read_parquet(bucket.get_object(key), **config)
        else:
            raise TypeError("Only .csv .parq or .parquet can be accessed")
    else:
        filepath = remove_prefix(url, "file://")
        if is_csv(filepath):
            return pd.read_csv(filepath, thousands=",", **config)
        elif is_parquet(filepath):
            return pd.read_parquet(filepath, **config)
        else:
            raise TypeError("Only .csv .parq or .parquet can be accessed")
