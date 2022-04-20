import shutil
from glob import glob
from anomal.enc_dec_ad.model import enc_dec_ad
from common.nn_detector import Detector as Predictor
from common.data_config import AnomalDataConfig, AnomalIndexerConfig
from common.data_utils import read_csv
import pandas as pd
from influxdb_client import InfluxDBClient
import random

random.seed(46)
kwargs = {
    "data_train": "file://data/air_passengers_train.csv",
    "data_val": "file://data/air_passengers_train.csv",
    "model_state": None,
    "loss": "MAE",
    "lr": 0.05,
    "batch_size": 8,
    "early_stopping_patience": 10,
    "weight_decay": None,
    "max_clip_grad_norm": None,
    "max_epochs": 1,
    "cell_type": "LSTM",
    "hidden_size": 32,
    "n_layers": 3,
    "dropout": 0.1,
    "sampler": True,
}
data_config = AnomalDataConfig(
    time="ds",
    freq="MS",
    indexer=AnomalIndexerConfig(name="slide_window_fixed", steps=10),
    group_ids=[],
    # categoricals=[Categorical(name= "cat", choices=["A", "B", "C", "D"])],
    # cat_features=["cat"],
    cont_features=["cont", "y"]
    # `unknown` will be added in dataset
)


def test_enc():
    # train
    predictor = Predictor(
        data_cfg=data_config,
        model_cls=enc_dec_ad,
        model_params={
            "cell_type": kwargs["cell_type"],
            "hidden_size": kwargs["hidden_size"],
            "n_layers": kwargs["n_layers"],
            "dropout": kwargs["dropout"],
        },
        loss_cfg=kwargs.get("loss", None),
        trainer_cfg={
            "batch_size": kwargs["batch_size"],
            "lr": kwargs["lr"],
            "early_stopping_patience": kwargs["early_stopping_patience"],
            "max_epochs": kwargs["max_epochs"],
            "sampler": kwargs["sampler"],
        },
    )
    data_train = read_csv(
        kwargs.pop("data_train"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_train["cont"] = data_train["y"].map(lambda x: x + random.random())

    data_val = read_csv(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_val["cont"] = data_val["y"].map(lambda x: x + random.random())
    predictor.train(data_train, data_val)
    assert len(glob(f"{predictor.root_dir}/*.pt")) > 0
    assert len(glob(f"{predictor.root_dir}/state.json")) > 0

    # validate
    data_val = read_csv(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_val["cont"] = data_val["y"].map(lambda x: x + random.random())
    predictor = Predictor.load_from_dir(predictor.root_dir, enc_dec_ad)
    metrics = predictor.validate(data_val)
    assert type(metrics) == dict

    # predict
    data_pre = read_csv(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_pre["cont"] = data_pre["y"].map(lambda x: x + random.random())
    predictor = Predictor.load_from_dir(predictor.root_dir, enc_dec_ad)
    result = predictor.predict(data_pre)
    assert [item in result.columns for item in ["ds", "y", "_time_idx_", "Anomaly_Score"]]
    assert result["Anomaly_Score"].count() == 10

    result.set_index("ds", inplace=True)
    result.index = pd.to_datetime(result.index)
    result["model"] = "enc_dec_ad"

    with InfluxDBClient(
        url="http://localhost:8086",
        token="m9nBYCOJ70_sSn5wDt9EyQfSSWDX4mjAGMt27-d2cF0d_BJsnRML5czj40_IOSW6IS1Uahm5eg0C2Io2QAmENw==",
        org="unianalysis",
        debug=True,
    ) as client:

        with client.write_api() as write_api:
            write_api.write(
                bucket="prediction-dev",
                record=result,
                data_frame_measurement_name="air_passenger_anomal",
                data_frame_tag_columns=["model"],
            )
            print("Wait to finishing ingesting DataFrame...")
        query = (
            'from(bucket:"prediction-dev")'
            " |> range(start: 0, stop: now())"
            ' |> filter(fn: (r) => r._measurement == "air_passenger_anomal")'
            ' |> filter(fn: (r) => r._field == "Anomaly_Score")'
        )
        result = client.query_api().query(query=query)

        client.delete_api().delete(
            bucket="prediction-dev",
            org="unianalysis",
            start="1900-01-02T23:00:00Z",
            stop="2022-01-02T23:00:00Z",
            predicate=' _measurement = "air_passenger_anomal" and model= "enc_dec_ad"',
        )

    shutil.rmtree(predictor.root_dir)


if __name__ == "__main__":
    test_enc()
