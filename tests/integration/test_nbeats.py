import shutil
from glob import glob
from ring.ts.nbeats.model import NbeatsNetwork
from ring.common.nn_predictor import Predictor
from ring.common.data_config import Categorical, DataConfig, IndexerConfig
from ring.common.data_utils import read_from_url
import pandas as pd
from influxdb_client import InfluxDBClient
import random

random.seed(46)
kwargs = {
    # "data_train": "file://data/air_passengers_train.csv",
    # "data_val": "file://data/air_passengers_train.csv",
    # "model_state": None,
    # "loss": "MAE",
    # "lr": 0.05,
    # "batch_size": 8,
    # "early_stopping_patience": 10,
    # "weight_decay": None,
    # "max_clip_grad_norm": None,
    # "max_epochs": 5,
    # "num_stack": 1,
    # "num_block": 3,
    # "width": 7,
    # "dropout": 0.1,
    # "expansion_coe": 5,
    # "backcast_loss_ratio": 0.1,
    "data_train": "file://data/parts_train.csv",
    "data_val": "file://data/parts_val.csv",
    "model_state": None,
    "loss": "MAE",
    "lr": 0.05,
    "batch_size": 8,
    "early_stopping_patience": 10,
    "weight_decay": None,
    "max_clip_grad_norm": None,
    "max_epochs": 30,
    "num_stack": 1,
    "num_block": 3,
    "width": 7,
    "dropout": 0.1,
    "expansion_coe": 5,
    "backcast_loss_ratio": 0.1,
}
data_config = DataConfig(
    # time="ds",
    # freq="MS",
    # targets=["y"],
    # indexer=IndexerConfig(name="slide_window", look_back=10, look_forward=5),
    # group_ids=[],
    # static_categoricals=[],
    # static_reals=[],
    # time_varying_known_categoricals=[],
    # time_varying_known_reals=[],
    # time_varying_unknown_categoricals=["cat"],
    # time_varying_unknown_reals=["cont"],
    # categoricals=[Categorical(name="cat", choices=["A", "B", "C", "D"])]
    # `unknown` will be added in dataset
    time="datetime",
    freq="MS",
    targets=["sales"],
    indexer=IndexerConfig(name="slide_window", look_back=10, look_forward=5),
    group_ids=["id"],
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=[],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
    categoricals=[],
)


def test_nbeats():
    # train
    predictor = Predictor(
        data_cfg=data_config,
        model_cls=NbeatsNetwork,
        model_params={
            "num_stack": kwargs["num_stack"],
            "num_block": kwargs["num_block"],
            "width": kwargs["width"],
            "expansion_coe": kwargs["expansion_coe"],
            "dropout": kwargs["dropout"],
            "backcast_loss_ratio": kwargs["backcast_loss_ratio"],
        },
        loss_cfg=kwargs.get("loss", None),
        trainer_cfg={
            "batch_size": kwargs["batch_size"],
            "lr": kwargs["lr"],
            "early_stopping_patience": kwargs["early_stopping_patience"],
            "max_epochs": kwargs["max_epochs"],
        },
    )
    data_train = read_from_url(
        kwargs.pop("data_train"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_train["cont"] = data_train["sales"].map(lambda x: x + random.random())

    data_val = read_from_url(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_val["cont"] = data_val["sales"].map(lambda x: x + random.random())
    predictor.train(data_train, data_val)
    assert len(glob(f"{predictor.save_dir}/*.pt")) > 0
    assert len(glob(f"{predictor.save_dir}/state.json")) > 0
    save_dir = predictor.save_dir
    # validate
    data_val = read_from_url(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_val["cont"] = data_val["sales"].map(lambda x: x + random.random())
    predictor = Predictor.load_from_dir(save_dir, NbeatsNetwork)
    metrics = predictor.validate(data_val)
    assert type(metrics) == dict

    # predict
    data_pre = read_from_url(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_pre["cont"] = data_pre["sales"].map(lambda x: x + random.random())
    predictor = Predictor.load_from_dir(save_dir, NbeatsNetwork)
    result = predictor.predict(data_pre, plot=False)
    # assert all([item in result.columns for item in ["ds", "y", "_time_idx_", "is_prediction", "y_pred"]])
    # assert result[~result["is_prediction"]]["y_pred"].isnull().all()
    # assert ~result[result["is_prediction"]]["y_pred"].isnull().any()
    # assert result[result["is_prediction"]]["y_pred"].count() == 5

    # result.set_index("ds", inplace=True)
    # result.index = pd.to_datetime(result.index)
    # result["model"] = "Nbeats"

    # with InfluxDBClient(
    #     url="http://localhost:8086",
    #     token="m9nBYCOJ70_sSn5wDt9EyQfSSWDX4mjAGMt27-d2cF0d_BJsnRML5czj40_IOSW6IS1Uahm5eg0C2Io2QAmENw==",
    #     org="unianalysis",
    #     debug=True,
    # ) as client:

    #     with client.write_api() as write_api:
    #         write_api.write(
    #             bucket="prediction-dev",
    #             record=result,
    #             data_frame_measurement_name="air_passenger_result",
    #             data_frame_tag_columns=["model"],
    #         )
    #         print("Wait to finishing ingesting DataFrame...")
    #     query = (
    #         'from(bucket:"prediction-dev")'
    #         " |> range(start: 0, stop: now())"
    #         ' |> filter(fn: (r) => r._measurement == "air_passenger_result")'
    #         ' |> filter(fn: (r) => r._field == "y_pred")'
    #     )
    #     result = client.query_api().query(query=query)

    #     client.delete_api().delete(
    #         bucket="prediction-dev",
    #         org="unianalysis",
    #         start="1900-01-02T23:00:00Z",
    #         stop="2022-01-02T23:00:00Z",
    #         predicate=' _measurement = "air_passenger_result" and model= "Nbeats"',
    #     )
    # shutil.rmtree(predictor.save_dir)


if __name__ == "__main__":
    test_nbeats()
