import torch
import collections.abc as collections
import itertools
import functools
from collections.abc import Mapping
from ignite.engine import Engine, DeterministicEngine
from ignite.metrics import Metric

from typing import Any, Dict, List, Union, Callable, Sequence, Type, Tuple, Optional, cast

from ring.common.loss import AbstractLoss, MAELoss
from ring.common.normalizers import AbstractNormalizer


def apply_to_type(
    x: Union[Any, collections.Sequence, collections.Mapping, str, bytes],
    input_type: Union[Type, Tuple[Type[Any], Any]],
    func: Callable,
) -> Union[Any, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on an object of `input_type` or mapping, or sequence of objects of `input_type`.

    Args:
        x: object or mapping or sequence.
        input_type: data type of ``x``.
        func: the function to apply on ``x``.
    """
    if isinstance(x, input_type):
        return func(x)
    if isinstance(x, (str, bytes)):
        return x
    if isinstance(x, collections.Mapping):
        return cast(Callable, type(x))(
            {k: apply_to_type(sample, input_type, func) for k, sample in x.items()}
        )
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return cast(Callable, type(x))(*(apply_to_type(sample, input_type, func) for sample in x))
    if isinstance(x, collections.Sequence):
        return cast(Callable, type(x))([apply_to_type(sample, input_type, func) for sample in x])
    if x is None:
        return None
    raise TypeError((f"x must contain {input_type}, dicts or lists; found {type(x)}"))


def apply_to_tensor(
    x: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes], func: Callable
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on a tensor or mapping, or sequence of tensors.

    Args:
        x: input tensor or mapping, or sequence of tensors.
        func: the function to apply on ``x``.
    """
    return apply_to_type(x, torch.Tensor, func)


def convert_tensor(
    x: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Move tensors to relevant device.

    Args:
        x: input tensor or mapping, or sequence of tensors.
        device: device type to move ``x``.
        non_blocking: convert a CPU Tensor with pinned memory to a CUDA Tensor
            asynchronously with respect to the host if possible
    """

    def _func(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, non_blocking=non_blocking) if device is not None else tensor

    return apply_to_tensor(x, _func)


def prepare_batch(
    batch: Sequence[torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training: pass to a device with options."""
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    gradient_accumulation_steps: int = 1,
):
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    n_parameters = [loss.n_parameters for loss in loss_fns]
    loss_end_indices = list(itertools.accumulate(n_parameters))
    loss_start_indices = [i - loss_end_indices[0] for i in loss_end_indices]

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)

        if isinstance(y_pred, Dict):
            try:
                y_forward = y_pred["prediction"]
                y_backcast = y_pred["backcast"][0]
                y_backcast_ratio = y_pred["backcast"][1]
            except:
                raise ValueError("output should have both `prediction` and `backcast`")

            reverse_scale_forward = lambda i, loss: loss.scale_prediction(
                y_forward[..., loss_start_indices[i] : loss_end_indices[i]],
                x["target_scales"][..., i],
                normalizers[i],
            )
            loss_forward = (
                functools.reduce(
                    lambda a, b: a + b,
                    [
                        loss_fn(reverse_scale_forward(i, loss_fn), y[..., i])
                        for i, loss_fn in enumerate(loss_fns)
                    ],
                )
                / len(loss_fns)
            )

            reverse_scale_backward = lambda i, loss: loss.scale_prediction(
                y_backcast[..., loss_start_indices[i] : loss_end_indices[i]],
                x["target_scales_back"][..., i],
                normalizers[i],
            )
            loss_backward = functools.reduce(
                lambda a, b: a + b,
                [
                    loss_fn(reverse_scale_backward(i, loss_fn), x["encoder_target"][..., i])
                    for i, loss_fn in enumerate(loss_fns)
                ],
            ) / len(
                loss_fns
            )  # mean loss of (target_i)
            loss = y_backcast_ratio * loss_backward + (1 - y_backcast_ratio) * loss_forward

        elif isinstance(y_pred, torch.Tensor):
            reverse_scale = lambda i, loss: loss.scale_prediction(
                y_pred[..., loss_start_indices[i] : loss_end_indices[i]],
                x["target_scales"][..., i],
                normalizers[i],
            )

            loss = (
                functools.reduce(
                    lambda a, b: a + b,
                    [loss_fn(reverse_scale(i, loss_fn), y[..., i]) for i, loss_fn in enumerate(loss_fns)],
                )
                / len(loss_fns)
            )
        else:
            raise TypeError("output of model must be one of torch.tensor or Dict")
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()

        return output_transform(x, y, y_pred, loss)

    return update


def supervised_evaluation_step(
    model: torch.nn.Module,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, y),
):
    n_parameters = [loss.n_parameters for loss in loss_fns]
    loss_end_indices = list(itertools.accumulate(n_parameters))
    loss_start_indices = [i - loss_end_indices[0] for i in loss_end_indices]

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)

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
                normalizers[i],
            )
            y_pred_scaled = torch.stack(
                [loss_obj.to_prediction(reverse_scale(i, loss_obj)) for i, loss_obj in enumerate(loss_fns)],
                dim=-1,
            )
            return output_transform(x, y, y_pred_scaled)

    return evaluate_step


def parameter_evaluation_step(
    model: torch.nn.Module,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
):
    n_parameters = [loss.n_parameters for loss in loss_fns]
    loss_end_indices = list(itertools.accumulate(n_parameters))
    loss_start_indices = [i - loss_end_indices[0] for i in loss_end_indices]
    error_vectors = []

    def evaluation_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        # calculate parametrs after training
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            reverse_scale = lambda i, loss: loss.scale_prediction(
                y_pred[..., loss_start_indices[i] : loss_end_indices[i]],
                x["target_scales"][..., i],
                normalizers[i],
            )
            y_pred_scaled = torch.stack(
                [loss_obj.to_prediction(reverse_scale(i, loss_obj)) for i, loss_obj in enumerate(loss_fns)],
                dim=-1,
            )
            error = MAELoss()(y_pred_scaled, y, reduce=None)
            # error_vectors += list(error.view(-1, y.shape[-1]).data.cpu().numpy())
            return error_vectors + list(error.view(-1, y.shape[-1]).data.cpu().numpy())

    return evaluation_step


def result_prediction_step(
    model: torch.nn.Module,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
):
    n_parameters = [loss.n_parameters for loss in loss_fns]
    loss_end_indices = list(itertools.accumulate(n_parameters))
    loss_start_indices = [i - loss_end_indices[0] for i in loss_end_indices]

    def prediction_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            reverse_scale = lambda i, loss: loss.scale_prediction(
                y_pred[..., loss_start_indices[i] : loss_end_indices[i]],
                x["target_scales"][..., i],
                normalizers[i],
            )
            y_pred_scaled = torch.stack(
                [loss_obj.to_prediction(reverse_scale(i, loss_obj)) for i, loss_obj in enumerate(loss_fns)],
                dim=-1,
            )
            error = MAELoss()(y_pred_scaled, y, reduce=None)
            return error, y_pred_scaled

    return prediction_step


###################
###################


def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
    gradient_accumulation_steps: int = 1,
) -> Engine:
    update_fn = supervised_training_step(
        model,
        optimizer,
        loss_fns,
        normalizers,
        device,
        non_blocking,
        prepare_batch,
        output_transform,
        gradient_accumulation_steps,
    )
    return Engine(update_fn) if not deterministic else DeterministicEngine(update_fn)


def create_supervised_evaluator(
    model: torch.nn.Module,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, y),
):
    evaluate_step = supervised_evaluation_step(
        model,
        loss_fns,
        normalizers,
        device,
        non_blocking,
        prepare_batch,
        output_transform,
    )

    evaluator = Engine(evaluate_step)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    return evaluator


def create_parameter_evaluator(
    model: torch.nn.Module,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
):
    evaluation_step = parameter_evaluation_step(
        model, loss_fns, normalizers, device, non_blocking, prepare_batch
    )

    evaluator = Engine(evaluation_step)
    return evaluator


def create_supervised_predictor(
    model: torch.nn.Module,
    loss_fns: List[AbstractLoss],
    normalizers: List[AbstractNormalizer],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = prepare_batch,
):
    prediction_step = result_prediction_step(
        model, loss_fns, normalizers, device, non_blocking, prepare_batch
    )

    predictor = Engine(prediction_step)
    return predictor
