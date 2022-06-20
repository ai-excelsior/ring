import math
import torch
from typing import Sequence, Union, Callable
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


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

        absolute_percentage_errors = torch.mean(2 * (y_pred - y).abs() / (y_pred.abs() + y.abs() + self._eps))

        self._sum_of_errors += absolute_percentage_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_symmetric_absolute_percentage_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "Symmetric Mean Absolute Percentage Error(SMAPE) must have at least one example before it can be computed."
            )
        return self._sum_of_errors.item() / self._num_examples * 100


class MAPE(Metric):
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

        mean_absolute_percentage_errors = torch.mean((y_pred - y).abs() / (y.abs() + self._eps))

        self._sum_of_errors += mean_absolute_percentage_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_mean_absolute_percentage_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "Mean Absolute Percentage Error(MAPE) must have at least one example before it can be computed."
            )
        return self._sum_of_errors.item() / self._num_examples * 100


class RMSE(Metric):
    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        root_mean_squared_errors = torch.mean(torch.pow(y_pred - y.view_as(y_pred), 2))

        self._sum_of_errors += root_mean_squared_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_root_mean_squared_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "Root Mean Squared Error(RMSE) must have at least one example before it can be computed."
            )
        return math.sqrt(self._sum_of_errors.item() / self._num_examples)


class MAE(Metric):
    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        mean_absolute_errors = torch.mean((y_pred - y).abs())

        self._sum_of_errors += mean_absolute_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_mean_absolute_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "Mean Absolute Error(MAE) must have at least one example before it can be computed."
            )
        return self._sum_of_errors.item() / self._num_examples


class MSE(Metric):
    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        mean_squared_errors = torch.mean(torch.pow(y_pred - y.view_as(y_pred), 2))

        self._sum_of_errors += mean_squared_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_mean_squared_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "Mean Squared Error(MSE) must have at least one example before it can be computed."
            )
        return self._sum_of_errors.item() / self._num_examples


class RSquare(Metric):
    @reinit__is_reduced
    def reset(self) -> None:
        self._mean_multi = torch.tensor(0.0, device=self._device)
        self._sum_y = torch.tensor(0.0, device=self._device)
        self._sum_y2 = torch.tensor(0.0, device=self._device)
        self._sum_ypred = torch.tensor(0.0, device=self._device)
        self._sum_ypred2 = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        # mean of steps
        mean_multi = torch.mean(torch.mul(y_pred, y.view_as(y_pred)))
        self._sum_y += torch.mean(y)
        self._sum_y2 += torch.mean(torch.pow(y, 2))
        self._sum_ypred += torch.mean(y_pred)
        self._sum_ypred2 += torch.mean(torch.pow(y_pred, 2))
        self._mean_multi += mean_multi
        self._num_examples += 1

    @sync_all_reduce("_sum_of_mean_squared_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "R Square (R2) must have at least one example before it can be computed."
            )

        return (self._num_examples * self._mean_multi - self._sum_ypred * self._sum_y) / (
            torch.pow(self._num_examples * self._sum_ypred2 - torch.pow(self._sum_ypred, 2), 0.5)
            * torch.pow(self._num_examples * self._sum_y2 - torch.pow(self._sum_y, 2), 0.5)
        )
