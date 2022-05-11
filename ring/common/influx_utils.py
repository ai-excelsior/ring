import os
import pandas as pd
from contextlib import contextmanager
from influxdb_client import InfluxDBClient


@contextmanager
def get_influx_client(*kwargs):
    if not kwargs:
        url = os.environ.get("INFLUX_ENDPOINT")
        token = os.environ.get("INFLUX_TOKEN")
        org = os.environ.get("INFLUX_ORG")
    else:
        url = kwargs.get("url")
        token = kwargs.get("token")
        org = kwargs.get("org")

    try:
        client = InfluxDBClient(
            url=url,
            token=token,
            org=org,
        )
        yield client
    finally:
        client.close()


def predictions_to_influx(
    df: pd.DataFrame,
    time_column: str,
    model_name: str,
    measurement: str,
    task_id: str = None,
):
    df.set_index(time_column, inplace=True)
    df.index = pd.to_datetime(df.index)
    df["model_name"] = model_name

    additional_tags = []
    if task_id is not None:
        df["task_id"] = task_id
        additional_tags.append("task_id")

    with get_influx_client() as client:
        with client.write_api() as write_api:
            write_api.write(
                bucket=os.environ.get("INFLUX_PREDICTION_BUCKET_NAME"),
                record=df,
                data_frame_measurement_name=measurement,
                data_frame_tag_columns=["model", "is_prediction", *additional_tags],
            )
