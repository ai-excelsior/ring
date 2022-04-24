import warnings
import pandas as pd
from sklearn.utils import shuffle
import torch
import os
import inspect
import numpy as np
import zipfile
import tempfile
import itertools
import shutil
from math import inf
from oss2 import Bucket
from glob import glob
from copy import deepcopy
from typing import Optional, Union, Dict, Any
from torch.utils.data.sampler import SubsetRandomSampler
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.handlers import Checkpoint, EarlyStopping, DiskSaver
from tabulate import tabulate
from .loss import MAELoss, SMAPELoss, RMSELoss, MAPELoss, MSELoss, cfg_to_losses
from .metrics import RMSE, SMAPE, MAE, MSE, MAPE
from .dataset_ano import TimeSeriesDataset
from .serializer import dumps, loads
from .utils import get_latest_updated_file, remove_prefix
from .trainer_utils import (
    create_supervised_trainer,
    prepare_batch,
    create_supervised_evaluator,
    create_parameter_evaluator,
    create_supervised_predictor,
)
from .data_config import DataConfig, dict_to_data_config_anomal
from .base_model import BaseModel
from .oss_utils import get_bucket_from_oss_url
from scipy.stats import multivariate_normal


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
        loss_cfg: str = "MSE",
        trainer_cfg: Dict = {},
        root_dir: str = None,
        device=None,
    ) -> None:
        """
        Initialize
        """
        self._losses = cfg_to_losses(loss_cfg, len(data_cfg.cont_features))
        model_params = deepcopy(model_params)
        model_params["output_size"] = sum([loss.n_parameters for loss in self._losses])

        os.makedirs(self.DEFAULT_ROOT_DIR, exist_ok=True)
        if root_dir is None:
            self.root_dir = tempfile.mkdtemp(prefix=self.DEFAULT_ROOT_DIR)
        else:
            self.root_dir = root_dir

        self._data_cfg = data_cfg
        self._trainer_cfg = trainer_cfg
        self._loss_cfg = loss_cfg
        self._model_cls = model_cls
        self._model_params = model_params

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
        return 1

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
        if isinstance(load, str) and not os.path.isfile(f"{self.root_dir}/{load}"):
            load = None
            warnings.warn(f"You are attemping to load file {load}, but it not exist.")

        # automatically get the last updated pt file
        if load == True:
            files = glob(f"{self.root_dir}{os.sep}*.pt")
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

        sampler = self._trainer_cfg.get("sampler", False) or self._model_cls.__name__ == "enc_dec_ad"
        batch_size = self._trainer_cfg.get("batch_size", 64)
        if self.enable_gpu:
            train_dataloader = dataset_train.to_dataloader(
                batch_size,
                num_workers=self.n_workers,
                shuffle=True,
                pin_memory=True,
                sampler=None,
            )
            if self._model_cls.__name__ != "enc_dec_ad":
                val_dataloader = dataset_val.to_dataloader(
                    batch_size,
                    train=False,
                    num_workers=self.n_workers,
                    sampler=SubsetRandomSampler if sampler else None,
                    ratio=self._model_params.get("train_gaussian_percentage", 0.25) if sampler else None,
                    pin_memory=True,
                )
            else:
                val_dataloader = dataset_val.to_dataloader(
                    batch_size,
                    train=False,
                    sampler=SubsetRandomSampler,
                    ratio=self._model_params.get("train_gaussian_percentage", 0.25),
                    num_workers=self.n_workers,
                    pin_memory=True,
                )
                gaussian_loader = dataset_train.to_dataloader(
                    batch_size,
                    train=False,
                    num_workers=self.n_workers,
                    sampler=SubsetRandomSampler,
                    ratio=self._model_params.get("train_gaussian_percentage", 0.25),
                    gaussian=True,
                )
        else:
            train_dataloader = dataset_train.to_dataloader(
                batch_size,
                num_workers=self.n_workers,
                shuffle=True,
                sampler=None,
            )
            if self._model_cls.__name__ != "enc_dec_ad":
                val_dataloader = dataset_val.to_dataloader(
                    batch_size,
                    train=False,
                    num_workers=self.n_workers,
                    sampler=SubsetRandomSampler if sampler else None,
                    ratio=self._model_params.get("train_gaussian_percentage", 0.25) if sampler else None,
                )
            else:
                val_dataloader = dataset_val.to_dataloader(  # for early_stop
                    batch_size,
                    train=False,
                    num_workers=self.n_workers,
                    sampler=SubsetRandomSampler,
                    ratio=self._model_params.get("train_gaussian_percentage", 0.25),
                )
                gaussian_loader = dataset_val.to_dataloader(  # for calculating parameters
                    batch_size,
                    train=False,
                    num_workers=self.n_workers,
                    sampler=SubsetRandomSampler,
                    ratio=self._model_params.get("train_gaussian_percentage", 0.25),
                    gaussian=True,
                )

        model = self.create_model(dataset_train)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._trainer_cfg.get("lr", 1e-3),
            weight_decay=self._trainer_cfg.get("weight_decay", 0),
        )

        trainer = create_supervised_trainer(
            model, optimizer, self._losses, normalizers=dataset_train._cont_scalars, device=self._device
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
            normalizers=dataset_train._cont_scalars,
            metrics=val_metrics,
            device=self._device,
        )

        gaussian_parameters = create_parameter_evaluator(
            model,
            loss_fns=self._losses,
            normalizers=dataset_train._cont_scalars,
            device=self._device,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(trainer):
            evaluator.run(val_dataloader)
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
                self.root_dir,
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
            if self._model_cls.__name__ == "enc_dec_ad":
                gaussian_parameters.run(gaussian_loader)
                output = gaussian_parameters.state.output
                mean = np.mean(output, axis=0)
                cov = np.cov(output, rowvar=False, dtype=mean.dtype)
                if not cov.shape:
                    cov = cov.reshape(1)
                self._model_params.update({"mean": mean, "cov": cov})
                print(f"Parameters for model Enc_Dec_AD is mean={mean}, covariance={cov}")

        # load
        if isinstance(load, str):
            Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(f"{self.root_dir}/{load}"))

        with TensorboardLogger(log_dir=f"{self.root_dir}") as logger:
            logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED(every=1),
                tag="train",
                output_transform=lambda loss: {"loss": loss},
            )
            logger.attach_output_handler(
                evaluator,
                event_name=Events.COMPLETED,
                tag="val",
                metric_names=list(val_metrics.keys()),
                global_step_transform=global_step_from_engine(trainer),
            )
            logger.attach_opt_params_handler(
                trainer, event_name=Events.ITERATION_STARTED, optimizer=optimizer
            )

            # evaluator.run(val_dataloader)
            trainer.run(train_dataloader, max_epochs=self._trainer_cfg.get("max_epochs", 200))
        self.save()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.
        """
        params = {
            name: getattr(self, f"_{name}", None) or getattr(self, name, None)
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["self", "_losses", "model_cls", "root_dir"]
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
            model_filename = get_last_updated_model(self.root_dir)

        model = self.create_model(dataset)
        Checkpoint.load_objects(
            to_load={"model": model}, checkpoint=torch.load(f"{self.root_dir}/{model_filename}")
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
            model, self._losses, dataset._cont_scalars, metrics=metrics, device=self._device
        )
        reporter.run(test_dataloader)
        headers = metrics.keys()  # metrics for simulating `look_forward` sequences
        print("===== Final Result =====")
        print(
            tabulate(
                [[reporter.state.metrics[key] for key in metrics.keys()]], headers=headers, tablefmt="tsv"
            )
        )

        return reporter.state.metrics

    def predict(
        self,
        data: pd.DataFrame = None,
        model_filename: str = None,
        **kwargs,
    ):
        """Do smoke test on given dataset, take the last max sequence to do a prediction and plot"""
        # use `last_only`=True to fetch only last `steps` result or `start_index` =  INT to assign detection start point

        dataset = self.create_dataset(data, **kwargs)

        # load model
        if model_filename is None:
            model_filename = get_last_updated_model(self.root_dir)
        model = self.create_model(dataset)
        Checkpoint.load_objects(
            to_load={"model": model}, checkpoint=torch.load(f"{self.root_dir}/{model_filename}")
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
            normalizers=dataset._cont_scalars,
            device=self._device,
        )
        scores = []

        @predictor.on(Events.ITERATION_COMPLETED)
        def record_score():
            if self._model_cls.__name__ == "enc_dec_ad":
                mvnormal = multivariate_normal(model.mean, model.cov, allow_singular=True)
                score = -mvnormal.logpdf(
                    predictor.state.output.view(-1, len(model._encoder_cont)).data.cpu().numpy()
                )
                scores.append(score.reshape(batch_size, self._data_cfg.indexer.steps))

        predictor.run(dataloader)
        scores = np.concatenate(scores)
        lattice = np.full((self._data_cfg.indexer.steps, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self._data_cfg.indexer.steps, i : i + self._data_cfg.indexer.steps] = score
        # combine all features to form `score` for this timestamps
        scores = np.nanmean(lattice, axis=0)

        raw_data = dataset.reflect(dataset._data.index)
        raw_data["Anomaly_Score"] = scores

        return raw_data

    @classmethod
    def from_parameters(cls, d: Dict, root_dir: str, model_cls: BaseModel) -> "Detector":
        self = cls(root_dir=root_dir, model_cls=model_cls, **d["params"])

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
    def load_from_dir(cls, root_dir: str, model_cls: BaseModel) -> Optional["Detector"]:
        """
        Load predictor from a dir
        """
        filepath = f"{root_dir}/state.json"

        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                state_dict = loads(f.read())
                state_dict["params"]["data_cfg"] = dict_to_data_config_anomal(
                    state_dict["params"]["data_cfg"]
                )
                return Detector.from_parameters(state_dict, root_dir, model_cls)

    @classmethod
    def load_from_oss_bucket(cls, bucket: Bucket, key: str, model_cls: BaseModel) -> Optional["Detector"]:
        if bucket.object_exists(key):
            root_dir = tempfile.mkdtemp(prefix=Detector.DEFAULT_ROOT_DIR)
            dest_zip_filepath = f"{root_dir}{os.sep}{key}"
            dirpath = os.path.dirname(dest_zip_filepath)
            os.makedirs(dirpath, exist_ok=True)
            bucket.get_object_to_file(key, dest_zip_filepath)
            zipfile.ZipFile(dest_zip_filepath).extractall(root_dir)
            os.remove(dest_zip_filepath)

            return cls.load_from_dir(root_dir, model_cls=model_cls)

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
            shutil.rmtree(self.root_dir)

    def save(self):
        """
        Save predictor's state to root_dir
        """
        parameters = self.get_parameters()
        with open(f"{self.root_dir}/state.json", "wb") as f:
            f.write(dumps(parameters))

    def zip(self, filepath: str = None):
        """zip last updated model file and state.json"""

        files = glob(f"{self.root_dir}{os.sep}*.pt")
        model_file = get_latest_updated_file(files)
        state_file = f"{self.root_dir}{os.sep}state.json"

        if filepath is None:
            filepath = os.path.join(self.root_dir, "model.zip")

        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_BZIP2) as archive:
            if model_file is not None:
                archive.write(model_file, os.path.basename(model_file))
            archive.write(state_file, os.path.basename(state_file))

        return filepath
