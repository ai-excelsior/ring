import functools
import os
from typing import Tuple
import oss2
import re

from ring.common.utils import remove_prefix


@functools.lru_cache(maxsize=64)
def get_bucket(bucket=None, endpoint=None):
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    bucket = bucket or os.environ.get("OSS_BUCKET_NAME")
    endpoint = endpoint or os.environ.get("OSS_ENDPOINT")

    return oss2.Bucket(oss2.Auth(key_id, key_secret), endpoint, bucket)


def parse_oss_url(url: str) -> Tuple[str, str, str]:
    """
    url format:  oss://{bucket}.{endpoint}/{key of the object}
    """
    url = remove_prefix(url, "oss://")
    return re.search(r"^([^.]*)\.([^/]*)/(.*)$", url).groups()


def get_bucket_from_oss_url(url: str):
    bucket, endpoint, key = parse_oss_url(url)
    return get_bucket(bucket, endpoint), key


@functools.lru_cache(maxsize=1)
def get_pandas_storage_options():
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = os.environ.get("OSS_ENDPOINT")

    return {
        "key": key_id,
        "secret": key_secret,
        "client_kwargs": {
            "endpoint_url": endpoint,
        },
        "config_kwargs": {"s3": {"addressing_style": "virtual"}},
    }
