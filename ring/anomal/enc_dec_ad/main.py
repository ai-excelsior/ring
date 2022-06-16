from argparse import ArgumentParser

import pandas as pd
from ring.anomal.enc_dec_ad.model import EncoderDecoderAD
from ring.common.cmd_parsers import (
    get_predict_parser,
    get_train_parser,
    get_validate_parser,
    get_serve_parser,
)
from ring.common.data_config import DataConfig, url_to_data_config_anomal
from ring.common.data_utils import read_from_url
from ring.common.influx_utils import predictions_to_influx
from ring.common.nn_detector import Detector as Predictor
import uvicorn
from fastapi import FastAPI


def train(data_config: DataConfig, data_train: pd.DataFrame, data_val: pd.DataFrame, **kwargs):
    model_state = kwargs.get("load_state", None)

    trainer_cfg = {
        "batch_size": kwargs["batch_size"],
        "lr": kwargs["lr"],
        "early_stopping_patience": kwargs["early_stopping_patience"],
        "max_epochs": kwargs["max_epochs"],
        "weight_decay": kwargs["weight_decay"],
    }

    predictor = None if model_state is None else Predictor.load(model_state, EncoderDecoderAD)
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
            },
            loss_cfg=kwargs.get("loss", None),
            trainer_cfg=trainer_cfg,
            save_dir=kwargs["save_state"],
            load_dir=kwargs["load_state"],
            num_workers=kwargs["num_workers"],
            logger_mode=kwargs["logger_mode"],
            task_id=kwargs["task_id"],
        )
        predictor.train(data_train, data_val)
    # first train
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

    predictor = Predictor.load(model_state, EncoderDecoderAD)
    predictor.validate(data_val)


def predict(
    load_state: str,
    data: pd.DataFrame,
    measurement: str = "prediction-dev",
    task_id: str = None,
):
    """
    load a model and predict with given dataset
    """
    assert load_state is not None, "load_state is required when validate"

    # predictor = Predictor.load(load_state, EncoderDecoderAD)
    # pred_df = predictor.predict(data, plot=False)
    from ring.common.data_utils import get_bucket_from_oss_url

    predictor = Predictor.load(load_state, EncoderDecoderAD)
    pred_df = predictor.predict(data, plot=False)
    bucket, key = get_bucket_from_oss_url(task_id)
    bucket.put_object_from_file(key, pred_df)


def serve(load_state, data_cfg):
    """
    load a model and predict with given dataset, using serve mode
    """
    # load_state = 4
    predictor = Predictor.load(load_state, EncoderDecoderAD)
    assert load_state is not None, "load_state is required when serve"
    data_cfg = url_to_data_config_anomal(data_cfg)

    app = FastAPI(
        title="Serve Predictor", description="API that load a trained model and do anomal detection"
    )

    @app.put("/")
    def read_data(data_json):
        parse_dates = [predictor._data_cfg.time] if data_cfg.time is None else [data_cfg.time]
        need_columns = (
            parse_dates
            + predictor._data_cfg.cont_features
            + predictor._data_cfg.cat_features
            + predictor._data_cfg.static_categoricals
            + predictor._data_cfg.group_ids
        )
        try:
            data = pd.read_json(
                data_json,
                orient="split",
                convert_dates=parse_dates,
                keep_default_dates=False,
                precise_float=True,
            )
        except:
            raise TypeError("read_json failed, please check your data, especially time column")
        # examine basic columns
        if [item for item in need_columns if item not in data.columns]:
            raise ValueError(
                f"columns don't match, because {[item for item in need_columns if item not in data.columns]} are needed, \
                    please check your data to make sure it matches the data config in trained model \
                        which are cont_features={predictor._data_cfg.cont_features}, \
                            cat_features={predictor._data_cfg.cat_features}, \
                            statistic_cat = {predictor._data_cfg.static_categoricals}, \
                                group_id={predictor._data_cfg.group_ids}, "
            )

        result = predictor.predict(data, plot=False)  # pd_df
        return result.to_json(orient="split", date_unit="ns", index=False)

    return app


if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = get_train_parser(subparsers)
    train_parser.add_argument(
        "--cell_type", type=str, choices=["LSTM", "GRU"], default="GRU", help="rnn cell type"
    )
    train_parser.add_argument("--hidden_size", type=int, default=32, help="hidden size of cell")
    train_parser.add_argument("--n_layers", type=int, default=1, help="layers of cell")
    train_parser.add_argument("--dropout", type=float, default=0.1)

    get_validate_parser(subparsers)
    get_predict_parser(subparsers)
    get_serve_parser(subparsers)

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
        uvicorn.run(serve(kwargs.pop("load_state", None), kwargs.pop("data_cfg")))
    else:
        raise ValueError("command should be one of train, validate, predict and serve")
