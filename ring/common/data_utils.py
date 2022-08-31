from tokenize import group
import pandas as pd
import tempfile
import os
from .oss_utils import get_bucket_from_oss_url
from .utils import remove_prefix


def is_parquet(s: str):
    return s.endswith(".parq") or s.endswith(".parquet")


def is_csv(s: str):
    return s.endswith(".csv")


def read_from_url(**config) -> pd.DataFrame:
    if is_parquet(config["url"]):
        config.pop("parse_dates")
        config.pop("dtype")

    if config["url"].startswith("oss://"):
        tempdir = tempfile.mkdtemp()
        bucket, key = get_bucket_from_oss_url(config.pop("url"))
        filename = f"{tempdir}/{key}"
        dirpath = os.path.dirname(filename)
        os.makedirs(dirpath, exist_ok=True)
        bucket.get_object_to_file(key, filename=filename)
    else:
        filename = remove_prefix(config.pop("url"), "file://")

    if is_csv(filename):
        config.update({"dtype": {k: object for k in config["dtype"]}})
        df = pd.read_csv(filename, thousands=",", **config)
    elif is_parquet(filename):
        df = pd.read_parquet(filename, **config)
    else:
        raise TypeError("Only .csv .parq or .parquet can be accessed")
    return df
