import pandas as pd
import tempfile
import os
from .oss_utils import get_bucket_from_oss_url
from .utils import remove_prefix


def is_parquet(s: str):
    return s.endswith(".parq") or s.endswith(".parquet")


def is_csv(s: str):
    return s.endswith(".csv")


def read_from_url(url: str, *args, **config) -> pd.DataFrame:
    if is_parquet(url):
        config.pop("parse_dates")
        config.pop("dtype")

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
        config.update({"dtype": {k: object for k in config["dtype"]}})
        df = pd.read_csv(filename, thousands=",", chunksize=1000, **config)
        if args[0] and args[1]:
            return pd.concat(
                [
                    chunk[
                        (chunk[config["parse_dates"][0]] >= args[0])
                        & (chunk[config["parse_dates"][0]] <= args[1])
                    ]
                    for chunk in df
                ]
            )
        elif args[0]:
            return pd.concat([chunk[(chunk[config["parse_dates"][0]] >= args[0])] for chunk in df])
        elif args[1]:
            return pd.concat([chunk[(chunk[config["parse_dates"][0]] <= args[1])] for chunk in df])
        else:
            return pd.concat([chunk for chunk in df])

    elif is_parquet(filename):
        df = pd.read_parquet(filename, **config)
        if args[0] and args[1]:
            return df[(df[config["parse_dates"][0]] >= args[0]) & (df[config["parse_dates"][0]] <= args[1])]
        elif args[0]:
            return df[df[config["parse_dates"][0]] >= args[0]]
        elif args[1]:
            return df[df[config["parse_dates"][0]] <= args[1]]
        else:
            return df

    else:
        raise TypeError("Only .csv .parq or .parquet can be accessed")


# iter_csv = pd.read_csv("file.csv", iterator=True, chunksize=1000)
# df = pd.concat([chunk[chunk["field"] > constant] for chunk in iter_csv])
