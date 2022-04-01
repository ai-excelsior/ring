from ring.ts.nbeats.model import NbeatsNetwork
from ring.ts.seq2seq.model import RNNSeq2Seq
from ring.common.nn_predictor import Predictor
from ring.common.data_config import DataConfig, IndexerConfig
from ring.common.data_utils import read_csv


kwargs = {
    "data_pre": "file://data/air_passengers_val.csv",
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


def test_predict_function_Nbeats():
    data_pre = read_csv(
        kwargs.pop("data_pre"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    tmpdir = kwargs.pop("model_state_nbeats", None)
    predictor = Predictor.load(tmpdir, NbeatsNetwork)
    result = predictor.smoke_test(data_pre, plot=True)
    assert [item in result.columns for item in ["ds", "y", "_time_idx_", "is_prediction", "y_pred"]]
    assert result[~result["is_prediction"]]["y_pred"].isnull().all()
    assert ~result[result["is_prediction"]]["y_pred"].isnull().any()
    assert result[result["is_prediction"]]["y_pred"].count() == 12


def test_predict_function_RNNseq2seq():
    data_pre = read_csv(
        kwargs.pop("data_pre"),
        parse_dates=[] if data_config.time is None else [data_config.time],
    )
    tmpdir = kwargs.pop("model_state_seq2seq", None)
    predictor = Predictor.load(tmpdir, RNNSeq2Seq)
    result = predictor.smoke_test(data_pre, plot=True)
    assert [item in result.columns for item in ["ds", "y", "_time_idx_", "is_prediction", "y_pred"]]
    assert result[~result["is_prediction"]]["y_pred"].isnull().all()
    assert ~result[result["is_prediction"]]["y_pred"].isnull().any()
    assert result[result["is_prediction"]]["y_pred"].count() == 12


if __name__ == "__main__":
    test_predict_function_RNNseq2seq()
