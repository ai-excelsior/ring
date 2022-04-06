import functools
import os
from typing import Tuple
import oss2
import re

from ring.common.utils import remove_prefix


@functools.lru_cache(maxsize=64)
def get_bucket(bucket, endpoint=None):
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = endpoint or os.environ.get("OSS_ENDPOINT")

    return oss2.Bucket(oss2.Auth(key_id, key_secret), endpoint, bucket)


def parse_oss_url(url: str) -> Tuple[str, str, str]:
    """
    url format:  oss://{bucket}/{key}
    """
    url = remove_prefix(url, "oss://")
    components = url.split("/")
    return components[0], "/".join(components[1:])


def get_bucket_from_oss_url(url: str):
    bucket_name, key = parse_oss_url(url)
    return get_bucket(bucket_name), key


@functools.lru_cache(maxsize=1)
def get_pandas_storage_options():
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = os.environ.get("OSS_ENDPOINT")

    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}"

    return {
        "key": key_id,
        "secret": key_secret,
        "client_kwargs": {
            "endpoint_url": endpoint,
        },
        # "config_kwargs": {"s3": {"addressing_style": "virtual"}},
    }
