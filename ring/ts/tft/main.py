import pandas as pd
from argparse import ArgumentParser
from ring.common.cmd_parsers import (
    get_predict_parser,
    get_train_parser,
    get_validate_parser,
    get_serve_parser,
)
from ring.common.data_config import DataConfig, url_to_data_config
from ring.common.nn_predictor import Predictor
from ring.common.influx_utils import predictions_to_influx
from ring.common.data_utils import read_from_url
from model import TemporalFusionTransformer
import uvicorn
from fastapi import FastAPI


def train(data_config: DataConfig, data_train: pd.DataFrame, data_val: pd.DataFrame, **kwargs):
    load_state = kwargs.get("load_state", None)

    trainer_cfg = {
        "batch_size": kwargs["batch_size"],
        "lr": kwargs["lr"],
        "early_stopping_patience": kwargs["early_stopping_patience"],
        "max_epochs": kwargs["max_epochs"],
        "weight_decay": kwargs["weight_decay"],
    }

    predictor = None if load_state is None else Predictor.load(load_state, TemporalFusionTransformer)
    if predictor is not None:
        predictor.trainer_cfg = trainer_cfg
        predictor.train(data_train, data_val, load=True)
    else:
        predictor = Predictor(
            data_cfg=data_config,
            model_cls=TemporalFusionTransformer,
            model_params={
                "hidden_size": kwargs["hidden_size"],
                "lstm_layers": kwargs["lstm_layers"],
                "dropout": kwargs["dropout"],
                "attention_head_size": kwargs["attention_head_size"],
                "max_encoder_length": kwargs["max_encoder_length"],
            },
            metric_cfg=kwargs.get("metric", None),
            trainer_cfg=trainer_cfg,
            save_dir=kwargs["save_state"],
            load_dir=kwargs["load_state"],
            num_workers=kwargs["num_workers"],
            logger_mode=kwargs["logger_mode"],
            task_id=kwargs["task_id"],
        )
        predictor.train(data_train, data_val)

    if load_state is None:
        print(f"Model saved in local file path: {predictor.save_dir}")
    #     if kwargs["save_state"].startswith("oss://"):
    #         predictor.upload(kwargs["save_state"])
    # else:
    #     predictor.upload(kwargs["save_state"])

    validate(kwargs.get("save_state", None), data_val, None)


def validate(load_state: str, data_val: pd.DataFrame, begin_point: str = None):
    """
    load a model and using this model to validate a given dataset
    """
    assert load_state is not None, "load_state is required when validate"

    predictor = Predictor.load(load_state, TemporalFusionTransformer)
    predictor.validate(data_val, begin_point=begin_point)


def predict(
    load_state: str,
    data: pd.DataFrame,
    measurement: str = "prediction-dev",
    task_id: str = None,
    begin_point: str = None,
):
    """
    load a model and predict with given dataset
    """
    assert load_state is not None, "load_state is required when validate"

    predictor = Predictor.load(load_state, TemporalFusionTransformer)
    pred_df = predictor.predict(data, plot=True, begin_point=begin_point)
    predictor.validate(data, begin_point=begin_point)

    predictions_to_influx(
        pred_df,
        time_column=predictor._data_cfg.time,
        model_name=predictor._model_cls.__module__,
        measurement=measurement,
        task_id=task_id,
        additional_tags=predictor._data_cfg.group_ids,
    )


def serve(load_state, data_cfg):
    """
    load a model and predict with given dataset, using serve mode
    """
    # TODO: should be modified to match new format
    # load_state = 4
    predictor = Predictor.load(load_state, TemporalFusionTransformer)
    assert load_state is not None, "load_state is required when serve"
    data_cfg = url_to_data_config(data_cfg)

    app = FastAPI(
        title="Serve Predictor", description="API that load a trained model and do anomal detection"
    )

    @app.put("/")
    def read_data(data_json):
        parse_dates = [predictor._data_cfg.time] if data_cfg.time is None else [data_cfg.time]
        need_columns = (
            parse_dates
            + predictor._data_cfg.targets
            + predictor._data_cfg.time_varying_known_categoricals
            + predictor._data_cfg.time_varying_unknown_categoricals
            + predictor._data_cfg.time_varying_known_reals
            + predictor._data_cfg.time_varying_unknown_reals
            + predictor._data_cfg.static_reals
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
                        which are targets={predictor._data_cfg.targets}, \
                            time_varying_known_categoricals={predictor._data_cfg.time_varying_known_categoricals}, \
                                time_varying_unknown_categoricals={predictor._data_cfg.time_varying_unknown_categoricals}, \
                                    time_varying_known_reals={predictor._data_cfg.time_varying_known_reals}, \
                                        time_varying_unknown_reals={predictor._data_cfg.time_varying_unknown_reals}, \
                            statistic_cat = {predictor._data_cfg.static_categoricals}, \
                                statistic_reals = {predictor._data_cfg.static_reals}, \
                                group_id={predictor._data_cfg.group_ids}, "
            )

        result = predictor.predict(data, plot=False)  # pd_df
        return result.to_json(orient="split", date_unit="ns", index=False)

    return app


if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    train_parser = get_train_parser(subparsers)
    train_parser.add_argument("--hidden_size", type=int, default=16, help="hidden size of cell")
    train_parser.add_argument("--lstm_layers", type=int, default=1, help="lstm_layers")
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--attention_head_size", type=int, default=4)
    train_parser.add_argument("--max_encoder_length", type=int, default=10)

    get_validate_parser(subparsers)
    get_predict_parser(subparsers)
    get_serve_parser(subparsers)

    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    if command == "train":
        assert 0 <= kwargs["dropout"] < 1, "dropout rate should be in the range of [0, 1)"

        data_config, data = url_to_data_config(
            kwargs.pop("data_cfg"),
            kwargs.pop("train_start_time"),
            kwargs.pop("train_end_time"),
            kwargs.pop("valid_start_time"),
            kwargs.pop("valid_end_time"),
        )
        train(data_config, data[0], data[1], **kwargs)

    elif command == "validate":
        data_config, data = url_to_data_config(
            kwargs.pop("data_cfg"),
            kwargs.pop("start_time"),
            kwargs.pop("end_time"),
        )
        validate(kwargs.pop("load_state", None), data, kwargs.pop("begin_point"))

    elif command == "predict":
        data_config, data = url_to_data_config(
            kwargs.pop("data_cfg"),
            kwargs.pop("start_time"),
            kwargs.pop("end_time"),
        )
        predict(
            kwargs.pop("load_state", None),
            data,
            measurement=kwargs.pop("measurement"),
            task_id=kwargs.pop("task_id", None),
            begin_point=kwargs.pop("begin_point"),
        )
    elif command == "serve":
        uvicorn.run(serve(kwargs.pop("load_state", None), kwargs.pop("data_cfg")))
    else:
        raise ValueError("command should be one of train, validate, predict and serve")
