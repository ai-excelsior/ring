import warnings
import pandas as pd
import torch
import os
import inspect
import numpy as np
import zipfile
import tempfile
from math import inf
import shutil
from oss2 import Bucket
from glob import glob
from copy import deepcopy
from typing import Optional, Union, Dict, Any
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.handlers import Checkpoint, EarlyStopping, DiskSaver
from .loss import cfg_to_losses
from .metrics import RMSE, SMAPE, MAE, MSE, MAPE
from .dataset_ano import TimeSeriesDataset
from .serializer import dumps, loads
from .utils import get_latest_updated_file, remove_prefix
from .trainer_utils import (
    create_supervised_trainer,
    create_supervised_evaluator,
    create_parameter_evaluator,
    create_supervised_predictor,
)
from .data_config import DataConfig, dict_to_data_config_anomal
from .base_model import BaseModel
from .oss_utils import get_bucket_from_oss_url
from .logger import Fluxlogger


def get_last_updated_model(filepath: str):
    files = glob(f"{filepath}{os.sep}*.pt")
    assert len(files) > 0, f"Can not find any .pt file in {filepath}, please make sure model is exist."

    list_of_models = get_latest_updated_file(files)
    return list_of_models.split(os.sep)[-1]


class Detector:
    DEFAULT_ROOT_DIR = "/tmp/"

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cls: BaseModel,
        model_params: Dict = {},
        model_states: Dict = {},
        loss_cfg: str = "MSE",
        trainer_cfg: Dict = {},
        save_dir: str = None,
        load_dir: str = None,
        device=None,
        num_workers=1,
        logger_mode: str = "local",
        task_id: str = "task_default_none",
    ) -> None:
        """
        Initialize
        """
        # make sure `cat_loss`` comes after `cont_loss`
        # because `dataset.target` = `dataset.cont` + `dataset.cat`
        if data_cfg.cat_features:
            if data_cfg.categoricals:  # already one-hot encoded
                self._losses = cfg_to_losses(loss_cfg, len(data_cfg.cont_features + data_cfg.cat_features))
            else:  # need to be encoded lately by dataset_ano
                self._losses = cfg_to_losses(loss_cfg, len(data_cfg.cont_features)) + cfg_to_losses(
                    "BCE", len(data_cfg.cat_features)
                )
        else:
            self._losses = cfg_to_losses(loss_cfg, len(data_cfg.cont_features))

        model_params = deepcopy(model_params)
        model_params["output_size"] = sum([loss.n_parameters for loss in self._losses])
        self._num_workers = num_workers
        os.makedirs(self.DEFAULT_ROOT_DIR, exist_ok=True)
        if save_dir is None or save_dir.startswith("oss://"):
            self.save_dir = tempfile.mkdtemp(prefix=self.DEFAULT_ROOT_DIR)
        else:
            self.save_dir = remove_prefix(save_dir, "file://")

        self.load_dir = load_dir

        self._data_cfg = data_cfg
        self._trainer_cfg = trainer_cfg
        self._loss_cfg = loss_cfg
        self._model_cls = model_cls
        self._model_params = model_params
        self._model_states = model_states
        self._logger = logger_mode
        self.task_id = task_id
        if device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device

        self._dataset_parameters = None
        # TODO: test
        print(f"Using device {self._device}")

    @property
    def trainer_cfg(self):
        return self._trainer_cfg

    @trainer_cfg.setter
    def trainer_cfg(self, trainer_cfg):
        self._trainer_cfg.update(trainer_cfg)

    @property
    def enable_gpu(self):
        if self._device == "cuda":
            return True
        return False

    @property
    def n_workers(self):
        if self.enable_gpu:
            return min(os.cpu_count() // 2, 2)
        return self._num_workers

    def create_dataset(self, data: pd.DataFrame, **kwargs):
        # if dataset_parameters exist we will always using this to initialize dataset
        if self._dataset_parameters is not None:
            return TimeSeriesDataset.from_parameters(self._dataset_parameters, data, **kwargs)

        dataset = TimeSeriesDataset.from_data_cfg(self._data_cfg, data)
        self._dataset_parameters = dataset.get_parameters()

        return dataset

    def create_model(self, dataset: TimeSeriesDataset) -> BaseModel:
        return self._model_cls.from_dataset(dataset, **self._model_params).to(self._device)

    def train(
        self,
        data_train: Union[pd.DataFrame, TimeSeriesDataset],
        data_val: Union[pd.DataFrame, TimeSeriesDataset],
        load: Union[bool, str] = False,
    ):
        """
        Train the model based on train_data and data_val
        """
        if isinstance(load, str) and not os.path.isfile(f"{self.load_dir}/{load}"):
            load = None
            warnings.warn(f"You are attemping to load file {load}, but it not exist.")

        # automatically get the last updated pt file
        if load is True:
            files = glob(f"{self.load_dir}{os.sep}*.pt")
            if len(files) == 0:
                load = None
            else:
                to_load = get_latest_updated_file(files)
                load = to_load.split(os.sep)[-1]

        if isinstance(data_train, TimeSeriesDataset):
            dataset_train = data_train
        else:
            dataset_train = self.create_dataset(data_train)

        if isinstance(data_val, TimeSeriesDataset):
            dataset_val = data_val
        else:
            dataset_val = self.create_dataset(data_val)

        batch_size = self._trainer_cfg.get("batch_size", 32)
        if self.enable_gpu:
            train_dataloader = dataset_train.to_dataloader(
                batch_size,
                num_workers=self.n_workers,
                shuffle=True,
                pin_memory=True,
            )
            val_dataloader = dataset_val.to_dataloader(
                batch_size,
                train=False,
                num_workers=self.n_workers,
                pin_memory=True,
            )
            gaussian_loader = dataset_val.to_dataloader(
                batch_size,
                train=False,
                num_workers=self.n_workers,
            )
        else:
            train_dataloader = dataset_train.to_dataloader(
                batch_size,
                num_workers=self.n_workers,
                shuffle=True,
            )
            val_dataloader = dataset_val.to_dataloader(  # for early_stop
                batch_size,
                train=False,
                num_workers=self.n_workers,
            )
            gaussian_loader = (
                dataset_val.to_dataloader(  # for calculating parameters, same as `val_dataloader`
                    batch_size,
                    train=False,
                    num_workers=self.n_workers,
                )
            )

        model = self.create_model(dataset_train)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._trainer_cfg.get("lr", 1e-3),
            weight_decay=self._trainer_cfg.get("weight_decay", 0),
        )

        trainer = create_supervised_trainer(
            model,
            optimizer,
            self._losses,
            normalizers=dataset_train._cont_scalars + dataset_train._categorical_encoders,
            device=self._device,
        )
        val_metrics = {
            "val_RMSE": RMSE(device=self._device),
            "val_MSE": MSE(device=self._device),
            "val_SMAPE": SMAPE(device=self._device),  # percentage
            "val_MAPE": MAPE(device=self._device),  # percentage
            "val_MAE": MAE(device=self._device),
        }
        evaluator = create_supervised_evaluator(
            model,
            self._losses,
            normalizers=dataset_train._cont_scalars + dataset_train._categorical_encoders,
            metrics=val_metrics,
            device=self._device,
        )

        gaussian_parameters = create_parameter_evaluator(
            model,
            loss_fns=self._losses,
            normalizers=dataset_train._cont_scalars + dataset_train._categorical_encoders,
            device=self._device,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(trainer):
            evaluator.run(val_dataloader, epoch_length=1000)
            metrics = evaluator.state.metrics
            print(
                f"Training Results - Epoch: {trainer.state.epoch}, {self._loss_cfg} Loss: {trainer.state.output:.2f}"
            )
            print(
                f"Val RMSE: {metrics['val_RMSE']:.2f},Val MSE: {metrics['val_MSE']:.2f},Val SMAPE: {metrics['val_SMAPE']:.2f},Val MAPE: {metrics['val_MAPE']:.2f},Val MAE: {metrics['val_MAE']:.2f}"
            )

        # checkpoint
        to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
        checkpoint = Checkpoint(
            to_save,
            save_handler=DiskSaver(
                self.save_dir,
                create_dir=True,
                require_empty=False,
            ),
            filename_prefix="best",
            score_function=lambda x: -x.state.metrics["val_" + self._loss_cfg]
            if "val_" + self._loss_cfg in x.state.metrics
            else -x.state.metrics["val_SMAPE"],
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            checkpoint,
        )

        # early stop
        early_stopping = EarlyStopping(
            patience=self._trainer_cfg.get("early_stopping_patience", 6),
            score_function=lambda engine: -engine.state.metrics["val_" + self._loss_cfg]
            if "val_" + self._loss_cfg in engine.state.metrics
            else -engine.state.metrics["val_RMSE"],
            min_delta=1e-8,
            trainer=trainer,
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            early_stopping,
        )

        # estimate parameters for `enc_dec_ad`
        @trainer.on(Events.COMPLETED)
        def evalutate_parameter():
            gaussian_parameters.run(gaussian_loader, epoch_length=1000)
            parameters = model.calculate_params(**gaussian_parameters.state.output)
            self._model_states.update(**parameters)
            for k, v in parameters.items():
                print(f"Parameters for model are {k}: {v}")

        # load
        if isinstance(load, str):
            Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(f"{self.load_dir}/{load}"))

        logger = (
            TensorboardLogger(log_dir=f"{self.save_dir}")
            if self._logger == "local"
            else Fluxlogger(task_id=f"{self.task_id}")
        )
        with logger:
            # train_itertaion loss
            logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED(every=1),
                tag="train_iteration",
                output_transform=lambda loss: {"loss": loss},
            )
            # train_epoch loss
            logger.attach_output_handler(
                trainer,
                event_name=Events.EPOCH_COMPLETED,
                tag="train_epoch",
                output_transform=lambda loss: {"loss": loss},
            )
            #  evaluate_epoch metric
            logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED(every=1),
                tag="validation_epoch",
                metric_names=list(val_metrics.keys()),
                global_step_transform=global_step_from_engine(trainer),
            )
            # optimizer_iteration
            logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                tag="optimizer_itertion",
                optimizer=optimizer,
            )
            trainer.run(
                train_dataloader, max_epochs=self._trainer_cfg.get("max_epochs", inf), epoch_length=1000
            )
        self.save()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.
        """
        params = {
            name: getattr(self, f"_{name}", None) or getattr(self, name, None)
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["self", "_losses", "model_cls"]
        }

        # pipeline and dataset
        dataset = self._dataset_parameters

        return dict(params=params, dataset=dataset)

    def validate(
        self,
        data_val: pd.DataFrame,
        model_filename=None,
    ):
        dataset = self.create_dataset(data_val)

        # load model
        if model_filename is None:
            #  `load_dir` not given
            if self.load_dir is None:
                # load model from last saved model
                self.load_dir = self.save_dir
            try:
                model_filename = get_last_updated_model(self.load_dir)
            except:
                raise ValueError("`load_state` shoule be provided")

        model = self.create_model(dataset)
        Checkpoint.load_objects(
            to_load={"model": model}, checkpoint=torch.load(f"{self.load_dir}/{model_filename}")
        )

        batch_size = self._trainer_cfg.get("batch_size", 64)
        metrics = {
            "RMSE": RMSE(device=self._device),
            "SMAPE": SMAPE(device=self._device),
            "MAE": MAE(device=self._device),
        }
        if self.enable_gpu:
            test_dataloader = dataset.to_dataloader(
                batch_size, train=False, num_workers=self.n_workers, pin_memory=True
            )
        else:
            test_dataloader = dataset.to_dataloader(batch_size, train=False, num_workers=self.n_workers)

        reporter = create_supervised_evaluator(
            model,
            self._losses,
            dataset._cont_scalars + dataset._categorical_encoders,
            metrics=metrics,
            device=self._device,
        )
        reporter.run(test_dataloader)
        # headers = metrics.keys()  # metrics for simulating `look_forward` sequences
        print("===== Final Result =====")
        print(str(dumps(reporter.state.metrics), "utf-8"))

        return reporter.state.metrics

    def predict(
        self,
        data: pd.DataFrame = None,
        model_filename: str = None,
        plot: str = False,
        **kwargs,
    ):
        """Do smoke test on given dataset, take all sequences by default"""
        # use `last_only`=True to fetch only last `steps` result or `start_index` =  INT to assign detection start point

        dataset = self.create_dataset(data, **kwargs)

        # load model
        if model_filename is None:
            #  `load_dir` not given
            if self.load_dir is None:
                # load model from last saved model
                self.load_dir = self.save_dir
            try:
                model_filename = get_last_updated_model(self.load_dir)
            except:
                raise ValueError("`load_state` shoule be provided")

        model = self.create_model(dataset)
        Checkpoint.load_objects(
            to_load={"model": model},
            checkpoint=torch.load(f"{self.load_dir}/{model_filename}", map_location=torch.device("cpu")),
        )

        batch_size = 1
        if self.enable_gpu:
            dataloader = dataset.to_dataloader(
                batch_size, train=False, num_workers=self.n_workers, pin_memory=True
            )
        else:
            dataloader = dataset.to_dataloader(batch_size, train=False, num_workers=self.n_workers)

        predictor = create_supervised_predictor(
            model,
            loss_fns=self._losses,
            normalizers=dataset._cont_scalars + dataset._categorical_encoders,
            device=self._device,
        )

        scores = []
        y_pred = []

        @predictor.on(Events.ITERATION_COMPLETED)
        def record_score():
            output = model.predict(predictor.state.output, **self._model_states)
            scores.append(output[0])
            y_pred.append(output[1].data.cpu().numpy())

        predictor.run(dataloader)
        scores = np.concatenate(scores)
        y_pred = np.concatenate(y_pred)
        lattice = np.full((self._data_cfg.indexer.steps, data.shape[0]), np.nan)
        lattice_ = np.full(
            (
                self._data_cfg.indexer.steps,
                data.shape[0],
                len(self._dataset_parameters["cont_feature"] + self._dataset_parameters["cat_feature"]),
            ),
            np.nan,
        )
        for i, score in enumerate(scores):
            lattice[i % self._data_cfg.indexer.steps, i : i + self._data_cfg.indexer.steps] = score
        # combine all features to form `score` for this timestamps
        scores = np.nanmean(lattice, axis=0)
        for i, out in enumerate(y_pred):
            lattice_[i % self._data_cfg.indexer.steps, i : i + self._data_cfg.indexer.steps, :] = out
        # combine all features to form `score` for this timestamps
        y_pred = np.nanmean(lattice_, axis=0)

        raw_data = dataset.reflect(dataset._data.index)
        raw_data["Anomaly_Score"] = scores
        # only output real and recon of `cont`
        prediction_column_names = [
            f"{target_name}_reconstruction"
            for target_name in self._dataset_parameters["cont_feature"]
            + self._dataset_parameters["cat_feature"]
        ]
        raw_data = raw_data.assign(**{name: np.nan for name in prediction_column_names})
        raw_data[prediction_column_names] = y_pred.reshape((-1, len(prediction_column_names)))
        return raw_data

    @classmethod
    def from_parameters(
        cls, d: Dict, save_dir: str, model_cls: BaseModel, new_save_dir: str = None
    ) -> "Detector":
        # dir to save this time
        d["params"]["save_dir"] = new_save_dir
        # dir to load from last time
        d["params"]["load_dir"] = save_dir
        self = cls(model_cls=model_cls, **d["params"])

        self._dataset_parameters = d["dataset"]

        return self

    @classmethod
    def from_cfg(cls) -> "Detector":
        """
        Construct a predictor from json config file.
        TODO
        """
        pass

    @classmethod
    def load_from_dir(
        cls, save_dir: str, model_cls: BaseModel, new_save_dir: str = None
    ) -> Optional["Detector"]:
        """
        Load predictor from a dir
        """
        filepath = f"{save_dir}/state.json"

        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                state_dict = loads(f.read())
                state_dict["params"]["data_cfg"] = dict_to_data_config_anomal(
                    state_dict["params"]["data_cfg"]
                )
                return Detector.from_parameters(state_dict, save_dir, model_cls, new_save_dir)

    @classmethod
    def load_from_oss_bucket(
        cls, bucket: Bucket, key: str, model_cls: BaseModel, new_save_dir: str = None
    ) -> Optional["Detector"]:
        if bucket.object_exists(key):
            save_dir = tempfile.mkdtemp(prefix=Detector.DEFAULT_ROOT_DIR)
            dest_zip_filepath = f"{save_dir}{os.sep}{key}"
            dirpath = os.path.dirname(dest_zip_filepath)
            os.makedirs(dirpath, exist_ok=True)
            bucket.get_object_to_file(key, dest_zip_filepath)
            zipfile.ZipFile(dest_zip_filepath).extractall(save_dir)
            os.remove(dest_zip_filepath)

            return cls.load_from_dir(save_dir, model_cls=model_cls, new_save_dir=new_save_dir)

    @classmethod
    def load(cls, url: str, model_cls: BaseModel) -> Optional["Detector"]:
        if url.startswith("file://"):
            return Detector.load_from_dir(remove_prefix(url, "file://"), model_cls)
        else:
            bucket, key = get_bucket_from_oss_url(url)
            return Detector.load_from_oss_bucket(bucket, key, model_cls)

    def upload(self, url: str):
        """upload model state to oss if given url is an oss file, else do nothing"""
        if url.startswith("oss://"):
            bucket, key = get_bucket_from_oss_url(url)
            zipfilepath = self.zip()
            bucket.put_object_from_file(key, zipfilepath)
            shutil.rmtree(self.save_dir)

    def save(self):
        """
        Save predictor's state to save_dir
        """
        parameters = self.get_parameters()
        with open(f"{self.save_dir}/state.json", "wb") as f:
            f.write(dumps(parameters))

    def zip(self, filepath: str = None):
        """zip last updated model file and state.json"""

        files = glob(f"{self.save_dir}{os.sep}*.pt")
        model_file = get_latest_updated_file(files)
        state_file = f"{self.save_dir}{os.sep}state.json"

        if filepath is None:
            filepath = os.path.join(self.save_dir, "model.zip")

        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_BZIP2) as archive:
            if model_file is not None:
                archive.write(model_file, os.path.basename(model_file))
            archive.write(state_file, os.path.basename(state_file))

        return filepath
