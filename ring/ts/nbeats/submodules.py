from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear(input_size, output_size, bias=True, dropout: int = None):
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


def linspace(
    backcast_length: int, forecast_length: int, centered: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class NBEATSBlock(nn.Module):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        share_thetas=False,
        dropout=0.1,
        tar_num=1,
        cov_num=0,
        tar_pos=[],
    ):

        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.tar_num = tar_num
        self.cov_num = cov_num
        self.tar_pos = tar_pos

        fc_stack = [
            [
                nn.Linear(backcast_length, units),
                nn.ReLU(),
            ]
            for i in range(self.tar_num + self.cov_num)
        ]

        for i in range(self.tar_num + self.cov_num):
            for _ in range(num_block_layers - 1):
                fc_stack[i].extend([linear(units, units, dropout=dropout), nn.ReLU()])

        self.fc = nn.ModuleList(nn.Sequential(*fc_stack[i]) for i in range(self.tar_num + self.cov_num))

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.ModuleList(
                [nn.Linear(units, thetas_dim, bias=False) for i in range(self.tar_num + self.cov_num)]
            )
        else:

            self.theta_b_fc = nn.ModuleList(
                [nn.Linear(units, thetas_dim, bias=False) for i in range(self.tar_num + self.cov_num)]
            )
            self.theta_f_fc = nn.ModuleList(
                [nn.Linear(units, thetas_dim, bias=False) for i in range(self.tar_num + self.cov_num)]
            )

    def forward(self, x):
        return torch.stack([self.fc[n](x[..., n]) for n in range(self.tar_num + self.cov_num)], dim=2)
        # return [self.fc[n](x[...,n]) for n in range(self.tar_num+self.cov_num)]


class NBEATSSeasonalBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim=None,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
        min_period=1,
        dropout=0.1,
        tar_num=1,
        cov_num=0,
        tar_pos=[],
    ):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length
        self.min_period = min_period

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
            tar_num=tar_num,
            cov_num=cov_num,
            tar_pos=tar_pos,
        )

        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=False)

        p1, p2 = (
            (thetas_dim // 2, thetas_dim // 2)
            if thetas_dim % 2 == 0
            else (thetas_dim // 2, thetas_dim // 2 + 1)
        )
        # seasonal_backcast_pre_p1  seasonal_backcast_pre_p2
        s1_b_pre = torch.tensor(
            [np.cos(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p1)],
            dtype=torch.float32,
        )  # H/2-1
        s2_b_pre = torch.tensor(
            [np.sin(2 * np.pi * i * backcast_linspace) for i in self.get_frequencies(p2)],
            dtype=torch.float32,
        )
        # concat seasonal_backcast_pre_p1 and seasonal_backcast_pre_p2
        s_b_pre = torch.stack(
            [torch.cat([s1_b_pre, s2_b_pre]) for n in range(self.tar_num + self.cov_num)],
            dim=2,
        )  # p1+p2 * backlength * tarnum

        # seasonal_forecast_pre_p1
        s1_f_pre = torch.tensor(
            [np.cos(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p1)],
            dtype=torch.float32,
        )
        # seasonal_forecast_pre_p2
        s2_f_pre = torch.tensor(
            [np.sin(2 * np.pi * i * forecast_linspace) for i in self.get_frequencies(p2)],
            dtype=torch.float32,
        )
        # concat seasonal_forecast_pre_p1 and seasonal_forecast_pre_p2
        s_f_pre = torch.stack(
            [torch.cat([s1_f_pre, s2_f_pre]) for n in range(self.tar_num + self.cov_num)],
            dim=2,
        )  # p1+p2 * forlength * tarnum

        # register, then can be applied as self.S_backcast and self.S_forecast
        self.register_buffer("S_backcast", s_b_pre)
        self.register_buffer("S_forecast", s_f_pre)
        self.agg_layer = nn.ModuleList([nn.Linear(tar_num + cov_num, 1) for i in range(tar_num)])

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)

        self.S_backcast_final = [self.agg_layer[i](self.S_backcast).squeeze(-1) for i in range(self.tar_num)]
        self.S_forecast_final = [self.agg_layer[i](self.S_forecast).squeeze(-1) for i in range(self.tar_num)]

        amplitudes_backward = torch.stack(
            [self.theta_b_fc[n](x[..., n]) for n in (self.tar_num + self.cov_num)], dim=2
        )

        backcast = torch.stack(
            [amplitudes_backward[..., n].mm(self.S_backcast_final) for n in self.tar_pos], dim=2
        )

        amplitudes_forward = torch.stack(
            [self.theta_f_fc[n](x[..., n]) for n in (self.tar_num + self.cov_num)], dim=2
        )
        forecast = torch.stack(
            [amplitudes_forward[..., n].mm(self.S_forecast_final) for n in self.tar_pos], dim=2
        )

        return backcast, forecast  # only target, not cov

    def get_frequencies(self, n):
        return np.linspace(0, (self.backcast_length + self.forecast_length) / self.min_period, n)


class NBEATSTrendBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
        tar_num=1,
        cov_num=0,
        tar_pos=[],
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            share_thetas=True,
            dropout=dropout,
            tar_num=tar_num,
            cov_num=cov_num,
            tar_pos=tar_pos,
        )

        backcast_linspace, forecast_linspace = linspace(backcast_length, forecast_length, centered=True)
        norm = np.sqrt(forecast_length / thetas_dim)  # ensure range of predictions is comparable to input

        # backcast
        coefficients_pre_b = torch.cat(
            [
                torch.tensor(
                    [backcast_linspace**i for i in range(thetas_dim)],
                    dtype=torch.float32,
                ).unsqueeze(2)
                for n in range(self.tar_num + self.cov_num)
            ],
            2,
        )
        # forecast
        coefficients_pre_f = torch.cat(
            [
                torch.tensor(
                    [forecast_linspace**i for i in range(thetas_dim)],
                    dtype=torch.float32,
                ).unsqueeze(2)
                for n in range(self.tar_num + self.cov_num)
            ],
            2,
        )

        # register, then can be applied as self.T_backcast and self.T_forecast
        self.register_buffer("T_backcast", coefficients_pre_b * norm)
        self.register_buffer("T_forecast", coefficients_pre_f * norm)

        self.agg_layer = nn.ModuleList([nn.Linear(tar_num + cov_num, 1) for i in range(tar_num)])

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().forward(x)

        self.T_backcast_final = [self.agg_layer[i](self.T_backcast).squeeze(-1) for i in range(self.tar_num)]
        self.T_forecast_final = [self.agg_layer[i](self.T_forecast).squeeze(-1) for i in range(self.tar_num)]

        backcast = torch.stack(
            [self.theta_b_fc[n](x[..., n]).mm(self.T_backcast_final[n]) for n in self.tar_pos], dim=2
        )
        forecast = torch.stack(
            [self.theta_f_fc[n](x[..., n]).mm(self.T_forecast_final[n]) for n in self.tar_pos], dim=2
        )
        return backcast, forecast


class NBEATSGenericBlock(NBEATSBlock):
    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
        tar_num=1,
        cov_num=0,
        tar_pos=[],
    ):

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            tar_num=tar_num,
            cov_num=cov_num,
            tar_pos=tar_pos,
        )

        self.backcast_fc = nn.ModuleList(
            [nn.Linear(thetas_dim, backcast_length) for i in range(self.tar_num)]
        )
        self.forecast_fc = nn.ModuleList(
            [nn.Linear(thetas_dim, forecast_length) for i in range(self.tar_num)]
        )

        self.agg_layer = nn.ModuleList([nn.Linear(tar_num + cov_num, 1) for i in range(tar_num)])

    def forward(self, x):
        x = super().forward(x)

        theta_bs = torch.cat(
            [F.relu(self.theta_b_fc[n](x[..., n])).unsqueeze(2) for n in range(self.tar_num + self.cov_num)],
            2,
        )  # encode x thetas_dim x n_output(n_tar+n_cov)
        theta_fs = torch.cat(
            [F.relu(self.theta_f_fc[n](x[..., n])).unsqueeze(2) for n in range(self.tar_num + self.cov_num)],
            2,
        )  # encode x thetas_dim x n_output

        # lengths = n_target
        theta_b = [self.agg_layer[i](theta_bs).squeeze(-1) for i in range(self.tar_num)]
        theta_f = [self.agg_layer[i](theta_fs).squeeze(-1) for i in range(self.tar_num)]

        return (
            torch.cat(
                [self.backcast_fc[i](theta_b[i]).unsqueeze(2) for i in range(self.tar_num)],
                2,
            ),
            torch.cat(
                [self.forecast_fc[i](theta_f[i]).unsqueeze(2) for i in range(self.tar_num)],
                2,
            ),
        )
