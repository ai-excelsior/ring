import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from torch import distributions
from typing import List, Dict
from collections import defaultdict
from .utils import register
from .normalizers import AbstractNormalizer
from .dtw import pairwise_distances, SoftDTW, PathDTW

LOSSES: Dict[str, "AbstractLoss"] = {}

# from dark to light
PREDICTION_COLORS = [
    "#8c2d04",
    "#d94801",
    "#f16913",
    "#fd8d3c",
    "#fdae6b",
    "#fdd0a2",
    "#fee6ce",
    "#fff5eb",
]


def _to_single_sequence(x: torch.Tensor):
    """
    Convert a 3d or 2d Tensor to 2d Tensor
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x[..., 0]

    raise ValueError(f"the prediction must be 2d or 3d, but got a {x.ndim}d.")


class AbstractLoss:
    @property
    def n_parameters(self):
        return 1

    @property
    def parameter_names(self):
        return ["pred"]

    def scale_prediction(
        self,
        y_pred: torch.Tensor,
        target_scale: torch.Tensor = None,
        normalizer: AbstractNormalizer = None,
        need=False,
    ):
        """
        rescale a predicted value from nearly [-1, 1] space to original space with scale and normalizer

        Args:
            y_pred (torch.Tensor): [batch_size, sequence_length, N], N is based on different losses
            scale (torch.Tensor, optional): [batch_size, sequence_length, 2]. Defaults to None. 2 means center&scale
            normalizer (AbstractNormalizer, optional): [description]. Defaults to None.
        """
        # convert to 2d tensor by default
        y_pred = y_pred[..., 0]
        # need to `reverse_transform`
        if need:
            if target_scale is None:
                return y_pred if normalizer is None else normalizer.inverse_postprocess(y_pred)
            # rescale back
            center = target_scale[..., 0]
            scale = target_scale[..., 1]
            y_scaled = y_pred * scale + center
            # if postprocess exist
            if normalizer:
                return normalizer.inverse_postprocess(y_scaled)

        return y_pred

    def to_prediction(self, y_pred: torch.Tensor, **kwargs):
        """
        Return single dimension prediction result

        Args:
            y_pred torch.Tensor: [batch_size, sequence_length]

        Returns:
            torch.Tensor
        """
        return y_pred

    def plot_one(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        x: str,
        target: str,
        title: str = "Prediction Plot - smoke testing",
        **_,
    ):
        # plot target
        (target_line,) = ax.plot(
            data[x],
            data[target],
            color="#4e79a7",
            label="time index",
        )
        # plot prediction
        (prediction_line,) = ax.plot(
            data[x],
            data[f"{target}_{self.parameter_names[0]}"],
            color=PREDICTION_COLORS[0],
            linestyle="--",
            label=target,
        )
        ax.legend([target_line, prediction_line], ["Ground Truth", "Prediction"])
        ax.set_xlabel("time index")
        ax.set_ylabel(target)
        ax.set_title(title)

    def plot(self, data: pd.DataFrame, x: str, target: str, group_ids: List[str] = [], n_samples=9, **kwargs):
        """Plot data with current loss function.

        The default behavior is plot line chart with the 1st `parameter_names` is prediction
        """
        data = data.sort_values([*group_ids, x])

        # sample groups
        if len(group_ids) > 0:
            # take last n_samples group
            g = data.groupby(group_ids)
            n_samples = min(n_samples, g.ngroups)
            ngroup = g.ngroup()

            fig, axes = plt.subplots(n_samples, 1, sharex=True, figsize=(9.6, 4.8 * n_samples), dpi=226)
            if n_samples == 1:
                group_info = [f"{group_id}={str(data.iloc[0][group_id])}" for group_id in group_ids]
                self.plot_one(axes, data, x, target, title=", ".join(group_info), **kwargs)
            else:
                for i, ax in enumerate(axes):
                    subset_data = data[ngroup == i]
                    group_info = [
                        f"{group_id}={str(subset_data.iloc[0][group_id])}" for group_id in group_ids
                    ]
                    self.plot_one(ax, subset_data, x, target, title=", ".join(group_info), **kwargs)
            return fig

        fig, ax = plt.subplots(sharex=True, figsize=(9.6, 4.8), dpi=226)
        self.plot_one(ax, data, x, target, **kwargs)
        return fig


@register(LOSSES)
class SMAPELoss(AbstractLoss):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        losses = (y_pred - y_true).abs() / (y_pred.abs() + y_true.abs() + 1e-8)
        return torch.mean(losses)


@register(LOSSES)
class MAPELoss(AbstractLoss):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        losses = (y_pred - y_true).abs() / (y_true.abs() + 1e-8)
        return torch.mean(losses)


@register(LOSSES)
class MAELoss(AbstractLoss):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, reduce="mean"):
        losses = (y_pred - y_true).abs()
        if reduce == "mean":
            return torch.mean(losses)
        elif reduce == "sum":
            return torch.sum(losses)
        else:
            return losses


@register(LOSSES)
class MSELoss(AbstractLoss):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        losses = torch.pow(y_pred - y_true, 2)
        return torch.mean(losses)


@register(LOSSES)
class RMSELoss(AbstractLoss):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        losses = torch.pow(y_pred - y_true, 2)
        return torch.sqrt(torch.mean(losses))


@register(LOSSES)
class BCELoss(AbstractLoss):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, reduce="mean"):
        losses = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
        if reduce == "mean":
            return torch.mean(losses)
        elif reduce == "sum":
            return torch.sum(losses)
        else:
            return losses


@register(LOSSES)
class DilateLoss(AbstractLoss):
    def __init__(self, gamma: float = 1, alpha: float = 0.5) -> None:
        super().__init__()

        self._alpha = alpha
        self._gamma = gamma

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Calculate dilate loss

        Args:
            y_pred (torch.Tensor): [batch_size, sequence_length]
            y_true (torch.Tensor): [batch_size, sequence_length]
        """
        sequence_length = y_pred.size(1)

        distances_matrix = pairwise_distances(y_true, y_pred)
        loss_shape = SoftDTW.apply(distances_matrix, self._gamma)

        path = PathDTW.apply(distances_matrix, self._gamma)
        omega = pairwise_distances(torch.arange(1, sequence_length + 1, dtype=torch.float).to(y_pred.device))
        loss_temporal = torch.sum(path * omega) / (sequence_length**2)

        return self._alpha * loss_shape + (1 - self._alpha) * loss_temporal


@register(LOSSES)
class QuantileLoss(AbstractLoss):
    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 0:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        else:
            quantiles = list(args)

        assert 0.5 in quantiles, "0.5 quantile is required in quantiles"
        assert all(
            [round(1 - q, 2) in quantiles for q in quantiles]
        ), "quantiles should be paired, for example (0.1, 0.9)"
        self._quantiles = sorted(quantiles)

    @property
    def n_parameters(self):
        return len(self._quantiles)

    @property
    def parameter_names(self):
        return [f"q{q}" for q in self._quantiles]

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_true = _to_single_sequence(y_true)

        losses = []
        for i, quantile in enumerate(self._quantiles):
            errors = y_true - y_pred[..., i]
            losses.append(torch.max((quantile - 1) * errors, quantile * errors))
        losses = torch.stack(losses, dim=-1)

        return torch.mean(losses)

    def scale_prediction(
        self, y_pred: torch.Tensor, target_scale: torch.Tensor = None, normalizer: AbstractNormalizer = None,need=False,
    ):
        if need:
            if target_scale is None:
                return y_pred if normalizer is None else normalizer.inverse_postprocess(y_pred)

            center = target_scale[..., 0].unsqueeze(-1)
            scale = target_scale[..., 1].unsqueeze(-1)
            y_scaled = y_pred * scale + center

            # if postprocess exist
            if normalizer:
                return normalizer.inverse_postprocess(y_scaled)

        return y_pred

    def to_prediction(self, y_pred: torch.Tensor, **kwargs):
        return y_pred[..., self._quantiles.index(0.5)]

    def plot_one(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        x: str,
        target: str,
        title: str = "Prediction Plot - smoke testing",
    ):
        # {column_name to color}
        color_map = {}
        for i, quantile_idx in enumerate(reversed(range(len(self._quantiles) // 2 + 1))):
            q = self._quantiles[quantile_idx]
            color_map[f"q{q}"] = PREDICTION_COLORS[i + 1]
            color_map[f"q{self._quantiles[len(self._quantiles) - i - 1]}"] = PREDICTION_COLORS[i + 1]

        # plot target
        (target_line,) = ax.plot(
            data[x],
            data[target],
            color="#4e79a7",
        )

        # plot quantile pairs
        quantile_patches = []
        quantile_labels = []
        for i in range(0, len(self._quantiles) // 2):
            q_lower_idx = i
            q_upper_idx = len(self._quantiles) - i - 1
            q_lower_name = self.parameter_names[q_lower_idx]
            q_upper_name = self.parameter_names[q_upper_idx]
            ax.fill_between(
                data[x],
                data[target + "_" + q_lower_name],
                data[target + "_" + q_upper_name],
                color=color_map[q_lower_name],
                alpha=0.6,
            )
            quantile_patches.append(mpatches.Patch(color=color_map[q_lower_name], alpha=0.6))
            quantile_labels.append(f"[{q_lower_name}, {q_upper_name}]")

        # plot prediction
        (prediction_line,) = ax.plot(
            data[x],
            data[target + "_" + "q0.5"],
            color=PREDICTION_COLORS[0],
            linestyle="--",
        )

        ax.legend(
            [target_line, prediction_line, *quantile_patches],
            ["Ground Truth", "Prediction", *quantile_labels],
        )
        ax.set_xlabel("time index")
        ax.set_ylabel(target)
        ax.set_title(title)


class DistributionLoss(AbstractLoss):
    """
    Using distribution as a loss function
    """

    distribution_class: distributions.Distribution = None
    distribution_arguments: List[str] = []

    @property
    def n_parameters(self):
        return len(self.distribution_arguments)

    @property
    def parameter_names(self):
        return self.distribution_arguments

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        distribution = self.to_distribution(y_pred)
        losses = -distribution.log_prob(_to_single_sequence(y_true))
        return torch.abs(torch.mean(losses))

    def to_distribution(self, y_pred: torch.Tensor) -> distributions.Distribution:
        raise NotImplementedError("implement this method")

    def scale_prediction(
        self, y_pred: torch.Tensor, target_scale: torch.Tensor = None, normalizer: AbstractNormalizer = None
    ):
        raise NotImplementedError("implement this method")

    def to_prediction(self, y_pred: torch.Tensor, use_metrics=True):
        if use_metrics:
            return self.to_distribution(y_pred).mean
        else:
            return y_pred.mean(-1)


@register(LOSSES)
class NormalDistrubutionLoss(DistributionLoss):
    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]

    def to_distribution(self, y_pred: torch.Tensor) -> distributions.Distribution:
        return self.distribution_class(y_pred[..., 0], y_pred[..., 1])

    def scale_prediction(
        self,
        y_pred: torch.Tensor,
        target_scale: torch.Tensor = None,
        normalizer: AbstractNormalizer = None,
        need=True,
    ):
        # TODO do I need consider normalizer transformation ?
        # rescale back
        if target_scale is not None:
            loc = y_pred[..., 0] * target_scale[..., 1] + target_scale[..., 0]
            scale = F.softplus(y_pred[..., 1]) * target_scale[..., 1]
        else:
            loc = y_pred[..., 0]
            scale = F.softplus(y_pred[..., 1])

        return torch.stack((loc, scale), dim=-1)

    def plot_one(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        x: str,
        target: str,
        title: str = "Prediction Plot - smoke testing",
        alpha=0.8,
    ):
        # plot target
        (target_line,) = ax.plot(
            data[x],
            data[target],
            color="#4e79a7",
            label="time index",
        )

        # plot confidence_interval
        confidence_interval = np.apply_along_axis(
            lambda x: stats.norm.interval(alpha, loc=x[0], scale=x[1]),
            1,
            data[[f"{target}_{name}" for name in self.parameter_names]],
        )
        ax.fill_between(
            data[x],
            confidence_interval[:, 0],
            confidence_interval[:, 1],
            color=PREDICTION_COLORS[1],
            alpha=0.6,
        )
        pred_interval_patch = mpatches.Patch(color=PREDICTION_COLORS[1], alpha=0.6)

        # plot prediction
        (prediction_line,) = ax.plot(
            data[x],
            data[f"{target}_{self.parameter_names[0]}"],
            color=PREDICTION_COLORS[0],
            linestyle="--",
            label=target,
        )

        ax.legend(
            [target_line, prediction_line, pred_interval_patch],
            ["Ground Truth", "Prediction", f"confid-{alpha}"],
        )
        ax.set_xlabel("time index")
        ax.set_ylabel(target)
        ax.set_title(title)


@register(LOSSES)
class NegativeBinomialDistrubutionLoss(DistributionLoss):
    distribution_class = distributions.NegativeBinomial
    distribution_arguments = ["mean", "shape"]

    def to_distribution(self, y_pred: torch.Tensor) -> distributions.Distribution:
        mean = y_pred[..., 0]
        shape = y_pred[..., 1]
        r = 1.0 / shape
        p = mean / (mean + r)
        return self.distribution_class(total_count=r, probs=p)

    def scale_prediction(
        self,
        y_pred: torch.Tensor,
        target_scale: torch.Tensor = None,
        normalizer: AbstractNormalizer = None,
        need=True,
    ):
        # TODO do I need consider normalizer transformation ?
        # rescale back
        if target_scale is not None:
            loc = F.softplus(y_pred[..., 0] * target_scale[..., 1] + target_scale[..., 0])
            scale = F.softplus(y_pred[..., 1]) / target_scale[..., 1].sqrt()
        else:
            loc = F.softplus(y_pred[..., 0])
            scale = F.softplus(y_pred[..., 1])

        return torch.stack((loc, scale), dim=-1)

    def plot_one(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        x: str,
        target: str,
        title: str = "Prediction Plot - smoke testing",
        alpha=0.8,
    ):
        # plot target
        (target_line,) = ax.plot(
            data[x],
            data[target],
            color="#4e79a7",
            label="time index",
        )

        # plot confidence_interval
        confidence_interval = np.apply_along_axis(
            lambda x: stats.nbinom.interval(
                alpha, n=np.round(1.0 / x[1]), p=x[0] / (x[0] + x[1] * (x[0] ** 2))
            ),
            1,
            data[[f"{target}_{name}" for name in self.parameter_names]],
        )
        ax.fill_between(
            data[x],
            confidence_interval[:, 0],
            confidence_interval[:, 1],
            color=PREDICTION_COLORS[1],
            alpha=0.6,
        )
        pred_interval_patch = mpatches.Patch(color=PREDICTION_COLORS[1], alpha=0.6)

        # plot prediction
        (prediction_line,) = ax.plot(
            data[x],
            np.round(data[f"{target}_{self.parameter_names[0]}"]),
            color=PREDICTION_COLORS[0],
            linestyle="--",
            label=target,
        )

        ax.legend(
            [target_line, prediction_line, pred_interval_patch],
            ["Ground Truth", "Prediction", f"confid-{alpha}"],
        )
        ax.set_xlabel("time index")
        ax.set_ylabel(target)
        ax.set_title(title)


def cfg_to_losses(cfg: str, n=1) -> List[AbstractLoss]:
    """
    convert config to loss class' instance, config looks like: 'Quantile,0.02,0.1,0.25,0.5,0.75,0.9,0.98'
    """
    cfg = cfg.split(",")
    loss_dict = defaultdict(lambda: "MSE")
    loss_dict.update(
        {
            "NbeatsNetwork": "SMAPE",
            "ReccurentNetwork": "MAE",
            "DeepAR": "NormalDistrubution",
            "TemporalFusionTransformer": "Quantile",
            "BCE": "BCE",
            "MAE": "MAE",
            "SMAPE": "SMAPE",
            "RMSE": "RMSE",
            "MAPE": "MAPE",
            "Dilate": "Dilate",
            "Quantile": "Quantile",
            "NormalDistrubution": "NormalDistrubution",
            "NegativeBinomialDistrubution": "NegativeBinomialDistrubution",
        }
    )
    loss_name = f"{loss_dict[cfg[0]]}Loss"
    loss_params = cfg[1:]
    params = []
    for param in loss_params:
        try:
            params.append(float(param))
        except:
            params.append(param)
    loss_cls = LOSSES[loss_name]
    return [loss_cls(*params) for _ in range(n)]
