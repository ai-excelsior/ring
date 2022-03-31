import tempfile
import pandas as pd
import zipfile
import os
import shutil
from argparse import ArgumentParser
from ring.common.cmd_parsers import get_predict_parser, get_train_parser, get_validate_parser
from ring.common.data_config import DataConfig, dict_to_data_config
from ring.common.serializer import loads
from ring.common.nn_predictor import Predictor
from ring.common.oss_utils import get_bucket
from ring.common.data_utils import read_csv
from model import NbeatsNetwork


def train(data_config: DataConfig, data_train: pd.DataFrame, data_val: pd.DataFrame, **kwargs):
    model_bucket = get_bucket()
    model_state = kwargs.get("model_state", None)

    is_load_from_model_state = model_state is not None and model_bucket.object_exists(model_state)
    if is_load_from_model_state:
        predictor = Predictor.load_from_oss_bucket(model_bucket, model_state, NbeatsNetwork)
        predictor.train(data_train, data_val, load=True)
    else:
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
        predictor.train(data_train, data_val)
    if model_state is None:
        print(f"Model saved in local file path: {predictor.root_dir}")
    else:
        zipfilepath = predictor.zip()
        model_bucket.put_object_from_file(model_state, zipfilepath)
        shutil.rmtree(predictor.root_dir)


def validate(model_state: str, data_val: pd.DataFrame):
    """
    load a model and using this model to validate a given dataset
    """
    model_bucket = get_bucket()

    assert model_state is not None, "model_state is required when validate"
    assert model_bucket.object_exists(model_state), "model_state should exist in oss bucket"

    predictor = Predictor.load_from_oss_bucket(model_bucket, model_state, NbeatsNetwork)
    predictor.validate(data_val)


def predict():
    pass


def serve():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = get_train_parser(subparsers)
    train_parser.add_argument("--num_stack", type=int, default=1)
    train_parser.add_argument("--num_block", type=int, default=3)
    train_parser.add_argument("--width", type=int, default=7)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--expansion_coe", type=int, default=5)
    train_parser.add_argument("--backcast_loss_ratio", type=float, default=0.1)

    get_validate_parser(subparsers)
    get_predict_parser(subparsers)

    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    if command == "train":
        assert 0 <= kwargs["dropout"] < 1, "dropout rate should be in the range of [0, 1)"
        # TODO: refactor this to support oss
        data_cfg_file = kwargs.pop("data_cfg")
        with open(data_cfg_file, "r") as f:
            data_config = dict_to_data_config(loads(f.read()))

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
        data_cfg_file = kwargs.pop("data_cfg")
        with open(data_cfg_file, "r") as f:
            data_config = dict_to_data_config(loads(f.read()))

        data_val = read_csv(
            kwargs.pop("data_val"),
            parse_dates=[] if data_config.time is None else [data_config.time],
        )
        validate(kwargs.pop("model_state", None), data_val)

    elif command == "predict":
        pass
    elif command == "serve":
        pass
