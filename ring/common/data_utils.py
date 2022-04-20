import pandas as pd
import tempfile
import os
from .oss_utils import get_bucket_from_oss_url
from .utils import remove_prefix


def is_parquet(s: str):
    return s.endswith(".parq") or s.endswith(".parquet")


def is_csv(s: str):
    return s.endswith(".csv")


def read_from_url(url: str, **config) -> pd.DataFrame:
    if is_parquet(url):
        config.pop("parse_dates")

    if url.startswith("oss://"):
        tempdir = tempfile.mkdtemp()
        bucket, key = get_bucket_from_oss_url(url)
        filename = f"{tempdir}/{key}"
        dirpath = os.path.dirname(filename)
        os.makedirs(dirpath, exist_ok=True)
        bucket.get_object_to_file(key, filename=filename)
    else:
        filename = remove_prefix(url, "file://")

    if is_csv(filename):
        return pd.read_csv(filename, thousands=",", **config)
    elif is_parquet(filename):
        return pd.read_parquet(filename, **config)
    else:
        raise TypeError("Only .csv .parq or .parquet can be accessed")
