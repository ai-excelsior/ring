import shutil
from glob import glob
from ring.ts.seq2seq.model import RNNSeq2Seq
from ring.common.nn_predictor import Predictor
from ring.common.data_config import DataConfig, IndexerConfig
from ring.common.data_utils import read_csv
import pandas as pd
from influxdb_client import InfluxDBClient

kwargs = {
    "data_train": "file://data/air_passengers_train.csv",
    "data_val": "file://data/air_passengers_train.csv",
    "model_state": None,
    "loss": "MAPE",
    "lr": 0.05,
    "batch_size": 8,
    "early_stopping_patience": 10,
    "weight_decay": None,
    "max_clip_grad_norm": None,
    "max_epochs": 1,
    "cell_type": "GRU",
    "hidden_size": 32,
    "n_layers": 1,
    "n_heads": 0,
    "dropout": 0.1,
}
data_config = DataConfig(
    time="ds",
    freq="MS",
    targets=["y"],
    group_ids=[],
    static_categoricals=[],
    indexer=IndexerConfig(
        name="slide_window", look_back=10, look_forward=5
    ),  # will still take model's parameters
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=[],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
)


def test_seq2seq():
    # train
    predictor = Predictor(
        data_cfg=data_config,
        model_cls=RNNSeq2Seq,
        model_params={
            "cell_type": kwargs["cell_type"],
            "hidden_size": kwargs["hidden_size"],
            "n_layers": kwargs["n_layers"],
            "dropout": kwargs["dropout"],
            "n_heads": kwargs["n_heads"],
        },
        loss_cfg=kwargs.get("loss", None),
        trainer_cfg={
            "batch_size": kwargs["batch_size"],
            "lr": kwargs["lr"],
            "early_stopping_patience": kwargs["early_stopping_patience"],
            "max_epochs": kwargs["max_epochs"],
        },
    )
    data_train = read_csv(
        kwargs.pop("data_train"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_val = read_csv(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    predictor.train(data_train, data_val)
    assert len(glob(f"{predictor.root_dir}/*.pt")) > 0
    assert len(glob(f"{predictor.root_dir}/state.json")) > 0

    # validate
    data_val = read_csv(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    predictor = Predictor.load_from_dir(predictor.root_dir, RNNSeq2Seq)
    predictor.validate(data_val)

    # predict
    data_pre = read_csv(
        kwargs.get("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    predictor = Predictor.load_from_dir(predictor.root_dir, RNNSeq2Seq)
    result = predictor.predict(data_pre, plot=False)
    assert [item in result.columns for item in ["ds", "y", "_time_idx_", "is_prediction", "y_pred"]]
    assert result[~result["is_prediction"]]["y_pred"].isnull().all()
    assert ~result[result["is_prediction"]]["y_pred"].isnull().any()
    assert result[result["is_prediction"]]["y_pred"].count() == 5

    result.set_index("ds", inplace=True)
    result.index = pd.to_datetime(result.index)
    result["model"] = "seq2seq"

    with InfluxDBClient(
        url="http://localhost:8086",
        token="m9nBYCOJ70_sSn5wDt9EyQfSSWDX4mjAGMt27-d2cF0d_BJsnRML5czj40_IOSW6IS1Uahm5eg0C2Io2QAmENw==",
        org="unianalysis",
        debug=True,
    ) as client:

        with client.write_api() as write_api:
            write_api.write(
                bucket="sample_result",
                record=result,
                data_frame_measurement_name="air_passenger_result",
                data_frame_tag_columns=["model"],
            )
            print("Wait to finishing ingesting DataFrame...")
        query = (
            'from(bucket:"sample_result")'
            " |> range(start: 0, stop: now())"
            ' |> filter(fn: (r) => r._measurement == "air_passenger_result")'
            ' |> filter(fn: (r) => r._field == "y_pred")'
        )
        result = client.query_api().query(query=query)

        client.delete_api().delete(
            bucket="sample_result",
            org="unianalysis",
            start="1900-01-02T23:00:00Z",
            stop="2022-01-02T23:00:00Z",
            predicate=' _measurement = "air_passenger_result" and model= "seq2seq"',
        )
    shutil.rmtree(predictor.root_dir)


# if __name__ == "__main__":
#     test_seq2seq()
