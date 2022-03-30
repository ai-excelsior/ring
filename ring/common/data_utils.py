import pandas as pd
from .oss_utils import get_pandas_storage_options


def read_csv(url: str, **config) -> pd.DataFrame:
    storage_option = None

    if url.startswith("oss://") or url.startswith("s3://"):
        url = url.replace("oss://", "s3://", 1)
        storage_option = get_pandas_storage_options()

    return pd.read_csv(url, thousands=",", storage_options=storage_option, **config)
