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

        root_mean_squared_errors = torch.mean(torch.sqrt(torch.pow(y_pred - y.view_as(y_pred), 2)))

        self._sum_of_errors += root_mean_squared_errors
        self._num_examples += 1

    @sync_all_reduce("_sum_of_absolute_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "MeanAbsoluteError must have at least one example before it can be computed."
            )
        return self._sum_of_errors.item() / self._num_examples
