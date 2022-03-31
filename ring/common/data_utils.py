import pandas as pd
from .oss_utils import get_bucket_from_oss_url
from .utils import remove_prefix


def read_csv(url: str, **config) -> pd.DataFrame:
    if url.startswith("oss://"):
        bucket, key = get_bucket_from_oss_url(url)
        return pd.read_csv(bucket.get_object(key), thousands=",")
    else:
        filepath = remove_prefix(url, "file://")
        return pd.read_csv(filepath, thousands=",", **config)
