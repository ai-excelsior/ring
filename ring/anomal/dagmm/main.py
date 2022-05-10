import pandas as pd
from argparse import ArgumentParser
from ring.common.cmd_parsers import get_predict_parser, get_train_parser, get_validate_parser
from ring.common.data_config import DataConfig, url_to_data_config_anomal
from ring.common.nn_detector import Detector as Predictor
from ring.common.influx_utils import predictions_to_influx
from ring.common.data_utils import read_from_url
from ring.anomal.dagmm.model import dagmm


def train(data_config: DataConfig, data_train: pd.DataFrame, data_val: pd.DataFrame, **kwargs):
    model_state = kwargs.get("load_state", None)

    trainer_cfg = {
        "batch_size": kwargs["batch_size"],
        "lr": kwargs["lr"],
        "early_stopping_patience": kwargs["early_stopping_patience"],
        "max_epochs": kwargs["max_epochs"],
        "sampler": True,
        "train_gaussian_percentage": kwargs["train_gaussian_percentage"],
    }

    predictor = None if model_state is None else Predictor.load(model_state, dagmm)

    if predictor is not None:
        predictor.trainer_cfg = trainer_cfg
        predictor.train(data_train, data_val, load=True)
    else:
        predictor = Predictor(
            data_cfg=data_config,
            model_cls=dagmm,
            model_params={
                "cell_type": kwargs["cell_type"],
                "hidden_size": kwargs["hidden_size"],
                "n_layers": kwargs["n_layers"],
                "dropout": kwargs["dropout"],
                "k_clusters": kwargs["k_clusters"],
                "encoderdecodertype": kwargs["encoderdecodertype"],
            },
            loss_cfg=kwargs.get("loss", None),
            trainer_cfg=trainer_cfg,
            save_dir=kwargs["save_state"],
            load_dir=kwargs["load_state"],
        )
        predictor.train(data_train, data_val)

    if model_state is None:
        print(f"Model saved in local file path: {predictor.save_dir}")
        if kwargs["save_state"].startswith("oss://"):
            predictor.upload(kwargs["save_state"])
    else:
        predictor.upload(kwargs["save_state"])


def validate(model_state: str, data_val: pd.DataFrame):
    """
    load a model and using this model to validate a given dataset
    """
    assert model_state is not None, "model_state is required when validate"

    predictor = Predictor.load(model_state, dagmm)
    predictor.validate(data_val)


def predict(
    model_state: str,
    data: pd.DataFrame,
    measurement: str = "prediction-dev",
    task_id: str = None,
):
    """
    load a model and predict with given dataset
    """
    assert model_state is not None, "model_state is required when validate"

    predictor = Predictor.load(model_state, dagmm)
    pred_df = predictor.predict(data, plot=True)
    predictions_to_influx(
        pred_df,
        time_column=predictor._data_cfg.time,
        model_name=predictor._model_cls.__module__,
        measurement=measurement,
        task_id=task_id,
    )


def serve():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = get_train_parser(subparsers)
    train_parser.add_argument("--cell_type", type=str, choices=["LSTM", "GRU"], default="LSTM")
    train_parser.add_argument("--hidden_size", type=int, default=5)
    train_parser.add_argument("--n_layers", type=int, default=1)
    train_parser.add_argument("--dropout", type=float, default=0)
    train_parser.add_argument("--k_clusters", type=float, default=3)
    train_parser.add_argument("--encoderdecodertype", type=str, default="RNN")

    get_validate_parser(subparsers)
    get_predict_parser(subparsers)

    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    if command == "train":
        assert 0 <= kwargs["dropout"] < 1, "dropout rate should be in the range of [0, 1)"

        data_config = url_to_data_config_anomal(kwargs.pop("data_cfg"))

        data_train = read_from_url(
            kwargs.pop("data_train"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        data_val = read_from_url(
            kwargs.pop("data_val"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        train(data_config, data_train, data_val, **kwargs)

    elif command == "validate":
        data_config = url_to_data_config_anomal(kwargs.pop("data_cfg"))

        data_val = read_from_url(
            kwargs.pop("data_val"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        validate(kwargs.pop("load_state", None), data_val)

    elif command == "predict":
        data_config = url_to_data_config_anomal(kwargs.pop("data_cfg"))

        data = read_from_url(
            kwargs.pop("data"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        predict(
            kwargs.pop("load_state", None),
            data,
            measurement=kwargs.pop("measurement"),
            task_id=kwargs.pop("task_id", None),
        )
    elif command == "serve":
        pass
