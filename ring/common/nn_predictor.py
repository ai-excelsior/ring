import warnings
import pandas as pd
import torch
import os
import inspect
import numpy as np
import zipfile
import tempfile
from oss2 import Bucket
from glob import glob
from copy import deepcopy
from typing import Union, Dict, Any
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.handlers import Checkpoint, EarlyStopping, DiskSaver
from tabulate import tabulate
from .loss import cfg_to_losses
from .metrics import RMSE, SMAPE
from .dataset import TimeSeriesDataset
from .serializer import dumps, loads
from .utils import add_time_idx, get_latest_updated_file
from .trainer_utils import create_supervised_trainer, prepare_batch, create_supervised_evaluator
from .data_config import DataConfig, dict_to_data_config
from .base_model import BaseModel


def get_last_updated_model(filepath: str):
    files = glob(f"{filepath}{os.sep}*.pt")
    assert len(files) > 0, f"Can not find any .pt file in {filepath}, please make sure model is exist."

    list_of_models = get_latest_updated_file(files)
    return list_of_models.split(os.sep)[-1]


class Predictor:
    DEFAULT_ROOT_DIR = "/tmp/ring/"

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
        self._losses = cfg_to_losses(loss_cfg, len(data_cfg.targets))
        model_params = deepcopy(model_params)
        model_params["output_size"] = sum([loss.n_parameters for loss in self._losses])

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
        data_train = add_time_idx(data_train, self._data_cfg.time, freq=self._data_cfg.freq)
        data_val = add_time_idx(data_val, self._data_cfg.time, freq=self._data_cfg.freq)

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

        self.save()

        batch_size = self._trainer_cfg.get("batch_size", 64)
        if self.enable_gpu:
            train_dataloader = dataset_train.to_dataloader(
                batch_size, num_workers=self.n_workers, pin_memory=True
            )
            val_dataloader = dataset_val.to_dataloader(
                batch_size, train=False, num_workers=self.n_workers, pin_memory=True
            )
        else:
            train_dataloader = dataset_train.to_dataloader(batch_size, num_workers=self.n_workers)
            val_dataloader = dataset_val.to_dataloader(batch_size, train=False, num_workers=self.n_workers)

        model = self.create_model(dataset_train)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._trainer_cfg.get("lr", 1e-3),
            weight_decay=self._trainer_cfg.get("weight_decay", 0),
        )

        trainer = create_supervised_trainer(
            model, optimizer, self._losses, normalizers=dataset_train.target_normalizers, device=self._device
        )
        val_metrics = {
            "val_rmse": RMSE(device=self._device),
            "val_smape": SMAPE(device=self._device),
        }
        evaluator = create_supervised_evaluator(
            model,
            self._losses,
            normalizers=dataset_train.target_normalizers,
            metrics=val_metrics,
            device=self._device,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(trainer):
            evaluator.run(val_dataloader)
            metrics = evaluator.state.metrics
            print(
                f"Training Results - Epoch: {trainer.state.epoch}, Loss: {trainer.state.output:.2f} Val RMSE: {metrics['val_rmse']:.2f}"
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
            score_function=lambda x: -x.state.metrics["val_rmse"],
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            checkpoint,
        )

        # early stop
        early_stopping = EarlyStopping(
            patience=self._trainer_cfg.get("early_stopping_patience", 6),
            score_function=lambda engine: -engine.state.metrics["val_rmse"],
            min_delta=1e-8,
            trainer=trainer,
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            early_stopping,
        )

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
        data_val = add_time_idx(data_val, self._data_cfg.time, freq=self._data_cfg.freq)
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
        }
        if self.enable_gpu:
            test_dataloader = dataset.to_dataloader(
                batch_size, train=False, num_workers=self.n_workers, pin_memory=True
            )
        else:
            test_dataloader = dataset.to_dataloader(batch_size, train=False, num_workers=self.n_workers)

        reporter = create_supervised_evaluator(
            model, self._losses, dataset.target_normalizers, metrics=metrics, device=self._device
        )
        reporter.run(test_dataloader)
        headers = metrics.keys()
        print("===== Final Result =====")
        print(
            tabulate(
                [[reporter.state.metrics[key] for key in metrics.keys()]], headers=headers, tablefmt="tsv"
            )
        )

        return reporter.state.metrics

    def smoke_test(
        self,
        data: pd.DataFrame,
        model_filename=None,
    ):
        """Do smoke test on given dataset, take the last max sequence to do a prediction and plot"""

        dataset = self.create_dataset(data)

        # load model
        if model_filename is None:
            model_filename = get_last_updated_model(self.root_dir)
        loss = self.create_loss()
        model = self.create_model(dataset, loss)
        Checkpoint.load_objects(
            to_load={"model": model}, checkpoint=torch.load(f"{self.root_dir}/{model_filename}")
        )

        # create predict mode dataset
        dataset = self.create_dataset(data, predict_mode=True)
        prediction_column_names = loss.parameter_names

        batch_size = len(dataset)
        if self.enable_gpu:
            dataloader = dataset.to_dataloader(
                batch_size, train=False, num_workers=self.n_workers, pin_memory=True
            )
        else:
            dataloader = dataset.to_dataloader(batch_size, train=False, num_workers=self.n_workers)

        model.eval()
        df = []
        with torch.no_grad():
            batch = next(iter(dataloader))
            x, y = prepare_batch(batch, self._device)
            y_pred = model(x)

            encoder_indices = x["encoder_idx"].detach().cpu().numpy().flatten().tolist()
            decoder_indices = x["decoder_idx"].detach().cpu().numpy().flatten().tolist()

            raw_data = dataset.reflect(encoder_indices, decoder_indices)
            raw_data = raw_data.assign(**{name: np.nan for name in prediction_column_names})
            raw_data.loc[decoder_indices, prediction_column_names] = (
                y_pred.reshape((-1, loss.n_parameters)).cpu().detach().numpy()
            )
            df.append(raw_data)
        df = pd.concat(df)

        # plot
        # 这里需要的，根据不同的loss，绘制对应target, group_ids的图像
        fig = loss.plot(raw_data, x=dataset._time_idx, target=dataset.targets, group_ids=dataset._group_ids)
        fig.savefig(f"{self.root_dir}{os.sep}smoke_testing.png")

    @classmethod
    def from_parameters(cls, d: Dict, root_dir: str, model_cls: BaseModel) -> "Predictor":
        self = cls(root_dir=root_dir, model_cls=model_cls, **d["params"])

        self._dataset_parameters = d["dataset"]

        return self

    @classmethod
    def from_cfg(cls) -> "Predictor":
        """
        Construct a predictor from json config file.
        TODO
        """
        pass

    @classmethod
    def load(cls, root_dir: str, model_cls: BaseModel) -> "Predictor":
        """
        Load predictor from a dir
        """
        filepath = f"{root_dir}/state.json"
        assert os.path.isfile(filepath)

        with open(filepath, "rb") as f:
            state_dict = loads(f.read())
            state_dict["params"]["data_cfg"] = dict_to_data_config(state_dict["params"]["data_cfg"])
            return Predictor.from_parameters(state_dict, root_dir, model_cls)

    @classmethod
    def load_from_oss_bucket(cls, bucket: Bucket, key: str, model_cls: BaseModel) -> "Predictor":
        root_dir = tempfile.mkdtemp(prefix=Predictor.DEFAULT_ROOT_DIR)
        dest_zip_filepath = f"{root_dir}{os.sep}{key}"
        bucket.get_object_to_file(key, dest_zip_filepath)
        zipfile.ZipFile(dest_zip_filepath).extractall(root_dir)
        os.remove(dest_zip_filepath)

        return cls.load(root_dir, model_cls=model_cls)

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
