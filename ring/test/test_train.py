from ring.ts.nbeats.model import NbeatsNetwork
from ring.ts.seq2seq.model import RNNSeq2Seq
from ring.common.nn_predictor import Predictor
from ring.common.data_config import DataConfig, url_to_data_config, IndexerConfig
from ring.common.data_utils import read_csv


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
    "num_stack": 1,
    "num_block": 3,
    "width": 7,
    "dropout": 0.1,
    "expansion_coe": 5,
    "backcast_loss_ratio": 0.1,
    "cell_type": "GRU",
    "hidden_size": 32,
    "n_layers": 1,
    "n_heads": 0,
}
data_config = DataConfig(
    time="ds",
    freq="MS",
    targets=["y"],
    indexer=IndexerConfig(name="slide_window", look_back=10, look_forward=5),
    group_ids=[],
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=[],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
)


def test_train_function_Nbeats():
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
    data_train = read_csv(
        kwargs.pop("data_train"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    data_val = read_csv(
        kwargs.pop("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    predictor.train(data_train, data_val)


def test_train_function_RNNseq2seq():
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
        kwargs.pop("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    predictor.train(data_train, data_val)


# if __name__ == "__main__":
#     test_train_function_RNNseq2seq()
