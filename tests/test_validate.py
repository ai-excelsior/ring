from ring.ts.nbeats.model import NbeatsNetwork
from ring.ts.seq2seq.model import RNNSeq2Seq
from ring.common.nn_predictor import Predictor
from ring.common.data_config import DataConfig, url_to_data_config, IndexerConfig
from ring.common.data_utils import read_csv


kwargs = {
    "data_val": "file://data/air_passengers.csv",
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
    "model_state_seq2seq": "file://example/seq2seq_model_state",
    "model_state_nbeats": "file://example/nbeats_model_state",
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


def test_validate_function_Nbeats():
    data_val = read_csv(
        kwargs.pop("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    predictor = Predictor.load(kwargs.pop("model_state_nbeats", None), NbeatsNetwork)
    predictor.validate(data_val)


def test_validate_function_RNNseq2seq():
    data_val = read_csv(
        kwargs.pop("data_val"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    predictor = Predictor.load(kwargs.pop("model_state_seq2seq", None), RNNSeq2Seq)
    predictor.validate(data_val)


# if __name__ == "__main__":
#     test_validate_function_Nbeats()
