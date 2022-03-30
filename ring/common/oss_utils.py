import functools
import os
import oss2


@functools.lru_cache(maxsize=1)
def get_model_bucket():
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    bucket = os.environ.get("OSS_BUCKET_NAME")
    endpoint = os.environ.get("OSS_ENDPOINT")

    return oss2.Bucket(oss2.Auth(key_id, key_secret), endpoint, bucket)


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
