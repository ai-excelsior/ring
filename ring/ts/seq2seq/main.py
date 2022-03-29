import pandas as pd
from argparse import ArgumentParser
from ring.common.cmd_parsers import get_predict_parser, get_train_parser, get_validate_parser
from ring.common.data_config import DataConfig, dict_to_data_config
from ring.common.serializer import loads
from ring.common.nn_predictor import Predictor
from model import RNNSeq2Seq

ROOT_DIR = "/tmp/ring"


def train(data_config: DataConfig, data_train: pd.DataFrame, data_val: pd.DataFrame, **kwargs):
    predictor = Predictor(
        data_cfg=data_config,
        model_cls=RNNSeq2Seq,
        model_params={},
        loss_cfg="MSE",
        trainer_cfg={"batch_size": 32, "early_stopping_patience": 12},
    )
    predictor.train(data_train=data_train, data_val=data_val)
    pass


def validate():
    pass


def predict():
    pass


def serve():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = get_train_parser(subparsers)

    get_validate_parser(subparsers)
    get_predict_parser(subparsers)

    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    if command == "train":
        data_cfg_file = kwargs.pop("data_cfg")
        with open(data_cfg_file, "r") as f:
            data_config = dict_to_data_config(loads(f.read()))

        data_train = pd.read_csv(
            kwargs.pop("data_train"),
            parse_dates=[] if data_config.time is None else [data_config.time],
            thousands=",",
        )
        data_val = pd.read_csv(
            kwargs.pop("data_val"),
            parse_dates=[] if data_config.time is None else [data_config.time],
            thousands=",",
        )

        train(data_config, data_train, data_val, **kwargs)
    elif command == "validate":
        pass
    elif command == "predict":
        pass
    elif command == "serve":
        pass
