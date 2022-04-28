from anomal.enc_dec_ad.model import enc_dec_ad
from common.nn_detector import Detector
from common.data_config import AnomalDataConfig, AnomalIndexerConfig
import pandas as pd
import random


def ecg_test():
    random.seed(46)
    ecg_data = pd.read_csv("data/ecg_time.csv")

    data_config = AnomalDataConfig(
        time=ecg_data.columns[0],
        freq="H",
        indexer=AnomalIndexerConfig(name="slide_window_fixed", steps=208),
        group_ids=["ids"],
        cont_features=["signal1", "signal2"],
    )
    kwargs = {
        "data_ecg": "file://data/ecg_time.csv",
        "model_state": None,
        "loss": "MAE",
        "lr": 0.05,
        "batch_size": 8,
        "early_stopping_patience": 10,
        "weight_decay": None,
        "max_clip_grad_norm": None,
        "max_epochs": 100,
        # config from paper
        "cell_type": "LSTM",
        "hidden_size": 45,
        "n_layers": 1,
        "dropout": 0,
        "sampler": True,
    }

    detector = Detector(
        data_cfg=data_config,
        model_cls=enc_dec_ad,
        model_params={
            "cell_type": kwargs["cell_type"],
            "hidden_size": kwargs["hidden_size"],
            "n_layers": kwargs["n_layers"],
            "dropout": kwargs["dropout"],
        },
        loss_cfg=kwargs.get("loss", None),
        trainer_cfg={
            "batch_size": kwargs["batch_size"],
            "lr": kwargs["lr"],
            "early_stopping_patience": kwargs["early_stopping_patience"],
            "max_epochs": kwargs["max_epochs"],
            "sampler": kwargs["sampler"],
        },
    )
    groups = ecg_data["ids"].unique()
    # train and test fetched by tag
    ecg_train_val = ecg_data[ecg_data["group"] == "train"].groupby("ids")
    ecg_test = ecg_data[ecg_data["group"] == "test"]
    # 70% as train, 30% as val
    ecg_train = pd.concat(
        [ecg_train_val.get_group(id)[: int(0.7 * len(ecg_train_val.get_group(id)))] for id in groups]
    )
    ecg_val = pd.concat(
        [ecg_train_val.get_group(id)[int(0.7 * len(ecg_train_val.get_group(id))) :] for id in groups]
    )
    # train
    detector.train(ecg_train.reset_index(drop=True), ecg_val.reset_index(drop=True))
    # validate, see metrics
    detector.validate(ecg_test)
    # predict, see scores
    detector.predict(ecg_test)


if __name__ == "__main__":
    ecg_test()
