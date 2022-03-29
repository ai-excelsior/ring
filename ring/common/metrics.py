import torch
from typing import Sequence, Union, Callable, Dict
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class Loss(Metric):
    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Loss, self).__init__(output_transform, device=device)
        self._loss_fn = loss_fn

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        average_loss = self._loss_fn(y_pred, y).detach()
        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        self._sum += average_loss.to(self._device)
        self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")
        return self._sum.item() / self._num_examples


class SMAPE(Metric):
    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = "cpu", eps=1e-8
    ):
        super().__init__(output_transform, device)
        self._eps = eps

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        absolute_errors = torch.mean(2 * (y_pred - y).abs() / (y_pred.abs() + y.abs() + self._eps))

        self._sum_of_errors += absolute_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_absolute_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "MeanAbsoluteError must have at least one example before it can be computed."
            )
        return self._sum_of_errors.item() / self._num_examples


class RMSE(Metric):
    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        root_mean_squared_errors = torch.mean(
            torch.sqrt(torch.sum(torch.pow(y_pred - y.view_as(y_pred), 2), dim=1))
        )

        self._sum_of_errors += root_mean_squared_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_absolute_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "MeanAbsoluteError must have at least one example before it can be computed."
            )
        return self._sum_of_errors.item() / self._num_examples
