from argparse import ArgumentParser

import pandas as pd
from ring.anomal.enc_dec_ad.model import EncoderDecoderAD
from ring.common.cmd_parsers import get_predict_parser, get_train_parser, get_validate_parser
from ring.common.data_config import DataConfig, url_to_data_config_anomal
from ring.common.data_utils import read_csv
from ring.common.influx_utils import predictions_to_influx
from ring.common.nn_detector import Detector as Predictor


def train(data_config: DataConfig, data_train: pd.DataFrame, data_val: pd.DataFrame, **kwargs):
    model_state = kwargs.get("model_state", None)

    trainer_cfg = {
        "batch_size": kwargs["batch_size"],
        "lr": kwargs["lr"],
        "early_stopping_patience": kwargs["early_stopping_patience"],
        "max_epochs": kwargs["max_epochs"],
        "sampler": kwargs[True],
    }

    if model_state is not None:
        predictor = Predictor.load(model_state, EncoderDecoderAD)

    if predictor is not None:
        predictor.trainer_cfg = trainer_cfg
        predictor.train(data_train, data_val, load=True)
    else:
        predictor = Predictor(
            data_cfg=data_config,
            model_cls=EncoderDecoderAD,
            model_params={
                "cell_type": kwargs["cell_type"],
                "hidden_size": kwargs["hidden_size"],
                "n_layers": kwargs["n_layers"],
                "dropout": kwargs["dropout"],
                "train_gaussian_percentage": kwargs["train_gaussian_percentage"],
            },
            loss_cfg=kwargs.get("loss", None),
            trainer_cfg=trainer_cfg,
        )
        predictor.train(data_train, data_val)

    if model_state is None:
        print(f"Model saved in local file path: {predictor.root_dir}")
    else:
        predictor.upload(model_state)


def validate(model_state: str, data_val: pd.DataFrame):
    """
    load a model and using this model to validate a given dataset
    """
    assert model_state is not None, "model_state is required when validate"

    predictor = predictor.load(model_state, EncoderDecoderAD)
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

    predictor = predictor.load(model_state, EncoderDecoderAD)
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
    train_parser.add_argument("--hidden_size", type=int, default=32)
    train_parser.add_argument("--n_layers", type=int, default=1)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train.parser.add_argument("--train_gaussian_percentage", type=float, default=0.25)

    get_validate_parser(subparsers)
    get_predict_parser(subparsers)

    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    if command == "train":
        assert 0 <= kwargs["dropout"] < 1, "dropout rate should be in the range of [0, 1)"

        data_config = url_to_data_config_anomal(kwargs.pop("data_cfg"))

        data_train = read_csv(
            kwargs.pop("data_train"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        data_val = read_csv(
            kwargs.pop("data_val"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        train(data_config, data_train, data_val, **kwargs)

    elif command == "validate":
        data_config = url_to_data_config_anomal(kwargs.pop("data_cfg"))

        data_val = read_csv(
            kwargs.pop("data_val"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        validate(kwargs.pop("model_state", None), data_val)

    elif command == "predict":
        data_config = url_to_data_config_anomal(kwargs.pop("data_cfg"))

        data = read_csv(
            kwargs.pop("data"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        predict(
            kwargs.pop("model_state", None),
            data,
            measurement=kwargs.pop("measurement"),
            task_id=kwargs.pop("task_id", None),
        )
    elif command == "serve":
        pass
