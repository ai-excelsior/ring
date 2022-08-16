import warnings
import pandas as pd
import torch
import os
import inspect
import numpy as np
import zipfile
import tempfile
import itertools
from math import inf
from oss2 import Bucket
from glob import glob
from copy import deepcopy
from typing import Optional, Union, Dict, Any
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.handlers import Checkpoint, EarlyStopping
from .loss import cfg_to_losses
from .metrics import RMSE, SMAPE, MAE, MSE, MAPE
from .dataset import TimeSeriesDataset
from .logger import Fluxlogger
from .oss_utils import DiskAndOssSaverAdd
from .serializer import dumps, loads
from .utils import get_latest_updated_file, remove_prefix
from .trainer_utils import create_supervised_trainer, prepare_batch, create_supervised_evaluator
from .data_config import DataConfig, dict_to_data_cfg
from .base_model import BaseModel
from .oss_utils import get_bucket_from_oss_url

PREDICTION_DATA = "_prediction_data_"


def get_last_updated_model(filepath: str):
    files = glob(f"{filepath}{os.sep}*.pt")
    assert len(files) > 0, f"Can not find any .pt file in {filepath}, please make sure model is exist."

    list_of_models = get_latest_updated_file(files)
    return list_of_models.split(os.sep)[-1]


class Predictor:
    DEFAULT_ROOT_DIR = "/tmp/"

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cls: BaseModel,
        model_params: Dict = {},
        metric_cfg: str = "MSE",
        trainer_cfg: Dict = {},
        save_dir: str = None,
        load_dir: str = None,
        device: str = None,
        num_workers=1,
        logger_mode: str = "local",
        task_id: str = "task_default_none",
    ) -> None:
        """
        Initialize
        """
        self._losses = cfg_to_losses(model_cls.__name__, len(data_cfg.targets))
        model_params = deepcopy(model_params)
        model_params["output_size"] = sum([loss.n_parameters for loss in self._losses])

        os.makedirs(self.DEFAULT_ROOT_DIR, exist_ok=True)
        # default model_save_dir
        if save_dir is None or save_dir.startswith("oss://"):
            self.save_dir = tempfile.mkdtemp(prefix=self.DEFAULT_ROOT_DIR)
            self.save_state = save_dir  # oss address
        else:
            self.save_dir = remove_prefix(save_dir, "file://")
            self.save_state = None

        self.load_dir = load_dir
        self._data_cfg = data_cfg
        self._trainer_cfg = trainer_cfg
        self._metric_cfg = metric_cfg
        self._model_params = model_params
        self._model_cls = model_cls
        self._num_workers = num_workers
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
            dataset_val = self.create_dataset(data_val, evaluate_mode=True)

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
            model,
            optimizer,
            self._losses,
            normalizers=dataset_train.target_normalizers,
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
            normalizers=dataset_train.target_normalizers,
            metrics=val_metrics,
            device=self._device,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(trainer):
            evaluator.run(val_dataloader)
            metrics = evaluator.state.metrics
            print(f"Training Results - Epoch: {trainer.state.epoch},  Loss: {trainer.state.output:.3f}")
            print(
                f"Val RMSE: {metrics['val_RMSE']:.3f},Val MSE: {metrics['val_MSE']:.3f},Val SMAPE: {metrics['val_SMAPE']:.3f},Val MAPE: {metrics['val_MAPE']:.3f},Val MAE: {metrics['val_MAE']:.3f}"
            )

        # checkpoint
        to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
        self.save()
        checkpoint = Checkpoint(
            to_save,
            save_handler=DiskAndOssSaverAdd(
                dirname=self.save_dir,
                ossaddress=self.save_state,
                create_dir=True,
                require_empty=False,
            ),
            filename_prefix="best",
            score_function=lambda x: -x.state.metrics["val_" + self._metric_cfg],
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            checkpoint,
        )

        # early stop
        early_stopping = EarlyStopping(
            patience=self._trainer_cfg.get("early_stopping_patience", 6),
            score_function=lambda engine: -engine.state.metrics["val_" + self._metric_cfg],
            min_delta=1e-8,
            trainer=trainer,
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            early_stopping,
        )

        # load
        if isinstance(load, str):
            Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(f"{self.load_dir}/{load}"))
        # record TRAIN:itertaion/epoch loss
        #       VALID:epoch/complete loss
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
                metric_names=list(val_metrics),
                global_step_transform=global_step_from_engine(trainer),
            )
            # optimizer_iteration
            logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                tag="optimizer_itertion",
                optimizer=optimizer,
            )
            trainer.run(train_dataloader, max_epochs=self._trainer_cfg.get("max_epochs", inf))

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

    def _examine_point(self, data: pd.DataFrame, begin_point: str) -> int:
        try:  # if int like str
            begin_point = data[
                data.index
                == (
                    data.index[0] + int(begin_point) - 1
                    if int(begin_point) > 0
                    else data.index[-1] + int(begin_point)
                )
            ].index.to_numpy()
        except:  # if datetime like str
            begin_point = data[data[self._data_cfg.time] == begin_point].index.to_numpy()
        finally:
            assert (
                begin_point and begin_point > 0 and begin_point < data.index[-1]
            ), "make sure begin_point is available in data"
            assert (
                begin_point >= data.index[0] + self._data_cfg.indexer.look_back - 1
            ), "not enough length for look_back"
            return int(begin_point)

    def verify_point(self, data: pd.DataFrame, begin_point: str) -> Union[Dict, int]:
        if not self._data_cfg.group_ids:
            data.name = PREDICTION_DATA
            return {PREDICTION_DATA: self._examine_point(data, begin_point)}
        else:
            return data.groupby(self._data_cfg.group_ids).apply(self._examine_point, begin_point).to_dict()

    def validate(
        self,
        data_val: pd.DataFrame,
        model_filename=None,
        begin_point: str = None,
    ):

        begin_point = self.verify_point(data_val, begin_point) if begin_point else begin_point
        assert (
            [
                idx
                <= data_val.groupby(self._data_cfg.group_ids).get_group(grp).index[-1]
                - self._data_cfg.indexer.look_forward
                for grp, idx in begin_point.items()
            ]
            if begin_point and self._data_cfg.group_ids
            else begin_point <= data_val.index[-1] - self._data_cfg.indexer.look_forward
            if begin_point
            else True
        ), "begin point should be not greater than last time point - look_forward in all groups"

        dataset = self.create_dataset(data_val, begin_point=begin_point, evaluate_mode=True)

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
            checkpoint=torch.load(
                f"{self.load_dir}/{model_filename}", map_location=torch.device(self._device)
            ),
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
            model, self._losses, dataset.target_normalizers, metrics=metrics, device=self._device
        )
        reporter.run(test_dataloader)
        print("===== Final Result =====")
        print(str(dumps(reporter.state.metrics), "utf-8"))

        return reporter.state.metrics

    def predict(
        self,
        data: pd.DataFrame,
        model_filename=None,
        begin_point: str = None,
        plot=False,
    ):
        """Do smoke test on given dataset, take the last max sequence to do a prediction and plot"""
        if len(self._data_cfg.time_varying_known_categoricals + self._data_cfg.time_varying_known_reals) > 0:
            begin_point = -self._data_cfg.indexer.look_forward  # last available point
        begin_point = self.verify_point(data, begin_point)
        # TODO: assert should consider limits
        assert (
            [
                idx <= data.groupby(self._data_cfg.group_ids).get_group(grp).index[-1]
                for grp, idx in begin_point.items()
            ]
            if self._data_cfg.group_ids
            else begin_point[data.name] <= data.index[-1]
        ), "begin point should be not greater than last time point in all groups"
        dataset = self.create_dataset(data, begin_point=begin_point, evaluate_mode=True, predict_task=True)

        # load model
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
            checkpoint=torch.load(
                f"{self.load_dir}/{model_filename}", map_location=torch.device(self._device)
            ),
        )

        # create predict mode dataset
        prediction_column_names = [f"{target_name}_pred" for i, target_name in enumerate(dataset.targets)]

        batch_size = len(dataset)
        if self.enable_gpu:
            dataloader = dataset.to_dataloader(
                batch_size, train=False, num_workers=self.n_workers, pin_memory=True
            )
        else:
            dataloader = dataset.to_dataloader(batch_size, train=False, num_workers=self.n_workers)

        model.eval()
        n_parameters = [loss.n_parameters for loss in self._losses]
        loss_end_indices = list(itertools.accumulate(n_parameters))
        loss_start_indices = [i - loss_end_indices[0] for i in loss_end_indices]
        with torch.no_grad():
            batch = next(iter(dataloader))
            x, y = prepare_batch(batch, self._device)
            y_pred = model(x, mode="predict")

            if isinstance(y_pred, Dict):
                try:
                    y_pred = y_pred["prediction"]
                except:
                    raise ValueError("output should have both `prediction` and `backcast`")
            elif not isinstance(y_pred, torch.Tensor):
                raise TypeError("output of model must be one of torch.tensor or Dict")

            reverse_scale = lambda i, loss: loss.scale_prediction(
                y_pred[..., loss_start_indices[i] : loss_end_indices[i]],
                x["target_scales"][..., i],
                dataset.target_normalizers[i],
                need=True,
            )
            y_pred_scaled = torch.stack(
                [
                    loss_obj.to_prediction(reverse_scale(i, loss_obj), use_metrics=False)
                    for i, loss_obj in enumerate(self._losses)
                ],
                dim=-1,
            )

            encoder_indices = x["encoder_idx"].detach().cpu().numpy().flatten().tolist()
            decoder_indices = x["decoder_idx"].detach().cpu().numpy().flatten().tolist()

            raw_data = dataset.reflect(encoder_indices, decoder_indices)
            raw_data = raw_data.assign(**{name: np.nan for name in prediction_column_names})

            # cuz `inverse_transform` can only deal with column names stored in state
            raw_data.loc[
                decoder_indices, prediction_column_names
            ] = dataset._target_detrenders.inverse_transform(
                pd.DataFrame(
                    y_pred_scaled.reshape((-1, len(prediction_column_names))).cpu().detach().numpy(),
                    columns=dataset.targets,
                    index=decoder_indices,
                ).join(raw_data.loc[decoder_indices, ["_time_idx_"] + dataset._group_ids]),
                dataset._group_ids,
            ).rename(
                lambda x: x + "_pred", axis=1
            )

        # plot
        # 这里需要的，根据不同的loss，绘制对应target, group_ids的图像
        if plot:
            # for deepar

            original_prediction_columns = list(
                filter(
                    lambda col: col not in prediction_column_names,
                    [
                        f"{target_name}_{param_name}"
                        for i, target_name in enumerate(dataset.targets)
                        for param_name in self._losses[i].parameter_names
                    ],
                )
            )
            if original_prediction_columns:
                raw_data = raw_data.assign(**{name: np.nan for name in original_prediction_columns})
                raw_data.loc[decoder_indices, original_prediction_columns] = (
                    torch.stack(
                        [reverse_scale(i, loss_obj) for i, loss_obj in enumerate(self._losses)],
                        dim=-1,
                    )
                    .reshape((-1, len(original_prediction_columns)))
                    .cpu()
                    .detach()
                    .numpy()
                )
            for i, loss in enumerate(self._losses):
                target_name = dataset.targets[i]
                fig = loss.plot(raw_data, x="_time_idx_", target=target_name, group_ids=dataset._group_ids)
                fig.savefig(f"{self.save_dir}{os.sep}smoke_testing_{target_name}.png")
            print(f"plotted figures saved at: {self.save_dir}")

        return raw_data

    @classmethod
    def from_parameters(
        cls, d: Dict, save_dir: str, model_cls: BaseModel, new_save_dir: str = None
    ) -> "Predictor":
        d["params"]["save_dir"] = new_save_dir
        d["params"]["load_dir"] = save_dir

        # remove unnecessary keys
        d["params"].pop("device", None)

        self = cls(model_cls=model_cls, **d["params"])

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
    def load_from_dir(
        cls, save_dir: str, model_cls: BaseModel, new_save_dir: str = None
    ) -> Optional["Predictor"]:
        """
        Load predictor from a dir
        """
        filepath = f"{save_dir}/state.json"

        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                state_dict = loads(f.read())
                state_dict["params"]["data_cfg"] = dict_to_data_cfg(state_dict["params"]["data_cfg"])
                return Predictor.from_parameters(state_dict, save_dir, model_cls, new_save_dir)

    @classmethod
    def load_from_oss_bucket(
        cls, bucket: Bucket, key: str, model_cls: BaseModel, new_save_dir: str = None
    ) -> Optional["Predictor"]:
        if bucket.object_exists(key):
            # create a random dir locally
            save_dir = tempfile.mkdtemp(prefix=Predictor.DEFAULT_ROOT_DIR)
            dest_zip_filepath = f"{save_dir}{os.sep}{key}"
            dirpath = os.path.dirname(dest_zip_filepath)
            os.makedirs(dirpath, exist_ok=True)
            # fetch oss files in dir
            bucket.get_object_to_file(key, dest_zip_filepath)
            zipfile.ZipFile(dest_zip_filepath).extractall(save_dir)
            os.remove(dest_zip_filepath)

            return cls.load_from_dir(save_dir, model_cls=model_cls, new_save_dir=new_save_dir)

    @classmethod
    def load(cls, url: str, model_cls: BaseModel, new_save_dir: str = None) -> Optional["Predictor"]:
        if url.startswith("file://"):
            return Predictor.load_from_dir(remove_prefix(url, "file://"), model_cls, new_save_dir)
        else:
            bucket, key = get_bucket_from_oss_url(url)
            return Predictor.load_from_oss_bucket(bucket, key, model_cls, new_save_dir)

    def save(self):
        """
        Save predictor's state to save_dir
        """
        parameters = self.get_parameters()
        with open(f"{self.save_dir}/state.json", "wb") as f:
            f.write(dumps(parameters))
