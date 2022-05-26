from ignite.contrib.handlers.base_logger import BaseLogger, BaseOutputHandler, BaseOptimizerParamsHandler
from ring.common.influx_utils import get_influx_client
from typing import Any, Optional, List, Callable, Union
from datetime import datetime
from ignite.engine import Engine, EventEnum, Events
from torch.optim import Optimizer
import os


class LoggerWriter(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.bucket = os.environ.get("INFLUX_LOG_BUCKET_NAME")

    def add_record(self, key, value, phase, event, task_id):
        record_dic = {
            "measurement": task_id,
            "tags": {"event": event, "phase": phase},
            "fields": {key: value},
            "time": datetime.utcnow().isoformat("T") + "Z",
        }
        with get_influx_client(**self.kwargs) as client:
            with client.write_api() as write_api:
                write_api.write(bucket=self.bucket, record=record_dic)


class Fluxlogger(BaseLogger):
    def __init__(self, task_id, **kwargs):
        self.writer = LoggerWriter(**kwargs)
        self.id = task_id

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(self.id, *args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(self.id, *args, **kwargs)

    # def close(self) -> None:
    #     self.writer.close()


class OutputHandler(BaseOutputHandler):
    def __init__(
        self,
        task_id: str,
        tag: str,
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable] = None,
        state_attributes: Optional[List[str]] = None,
    ):
        super(OutputHandler, self).__init__(
            tag, metric_names, output_transform, global_step_transform, state_attributes
        )
        self.task_id = task_id

    def __call__(self, engine: Engine, logger: Fluxlogger, event_name: Union[str, EventEnum]) -> None:

        if not isinstance(logger, Fluxlogger):
            raise RuntimeError("Handler 'OutputHandler' works only with Fluxlogger")

        metrics = self._setup_output_metrics_state_attrs(engine, key_tuple=False)

        if self.tag.startswith("train"):
            phase = "training"
            event = "iteration" if self.tag.endswith("iteration") else "epoch"
        elif self.tag.startswith("validation"):
            phase = "validation"
            event = "epoch"
        else:
            raise ValueError("tag should be start with train or validation")

        global_step = self.global_step_transform(engine, event_name)  # type: ignore[misc]
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        for key, value in metrics.items():
            logger.writer.add_record(key.split("/")[1], value, phase=phase, event=event, task_id=self.task_id)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    def __init__(self, task_id: str, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)
        self.task_id = task_id

    def __call__(self, engine: Engine, logger: Fluxlogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, Fluxlogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with Fluxlogger")

        # global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}" if self.tag else "None_specified"
        params = {
            f"{tag_prefix}/{self.param_name}/group_{i}": float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        for k, v in params.items():
            logger.writer.add_record(
                k.split("/")[1], v, phase="training", event="iteration", task_id=self.task_id
            )
