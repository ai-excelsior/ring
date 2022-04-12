import pandas as pd
from .oss_utils import get_bucket_from_oss_url
from .utils import remove_prefix


def read_csv(url: str, **config) -> pd.DataFrame:
    if url.startswith("oss://"):
        bucket, key = get_bucket_from_oss_url(url)
        if bucket.get_object(key)[-4:] == ".csv":
            return pd.read_csv(bucket.get_object(key), thousands=",")
        elif bucket.get_object(key)[-5:] == ".parq":
            return pd.read_parquet(bucket.get_object(key))
        else:
            raise TypeError("Only .csv or .parq can be accessed")
    else:
        filepath = remove_prefix(url, "file://")
        if filepath[-4:] == ".csv":
            return pd.read_csv(filepath, thousands=",", **config)
        elif filepath[-5:] == ".parq":
            return pd.read_parquet(filepath)
        else:
            raise TypeError("Only .csv or .parq can be accessed")
