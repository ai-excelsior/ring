import torch
from torch import nn
from typing import List, Dict, Tuple
from copy import deepcopy
from ring.common.base_model import AutoRegressiveBaseModelWithCovariates
from ring.common.dataset import TimeSeriesDataset


class ReccurentNetwork(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        targets: List[str] = [],
        output_size: int = 1,
        # hpyerparameters
        cell_type: str = "GRU",
        hidden_size: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        # data types
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        decoder_cont: List[str] = [],
        decoder_cat: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        x_categoricals: List[str] = [],
        target_lags: Dict = {},
    ):

        super().__init__(
            targets=targets,
            cell_type=cell_type,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            encoder_cont=encoder_cont,
            encoder_cat=encoder_cat,
            decoder_cont=decoder_cont,
            decoder_cat=decoder_cat,
            embedding_sizes=embedding_sizes,
            x_categoricals=x_categoricals,
            target_lags=target_lags,
            output_size=output_size,
        )
        self.output_projector_decoder = (
            nn.Linear(hidden_size, output_size)
            if len(self._targets) == 1
            else nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(output_size)])
        )

    def forward(self, x: Dict[str, torch.Tensor], **kwargs):
        hidden_state = self.encode(x)
        first_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].shape[0], device=x["encoder_cont"].device),
            x["encoder_length"] - 1,
            self.target_positions.unsqueeze(-1),
        ].t()
        prediction = self.decode(x, hidden_state=hidden_state, first_target=first_target)
        return prediction

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "ReccurentNetwork":
        # update embedding sizes from kwargs
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = {}
        for k, v in dataset.embedding_sizes.items():
            if k in dataset.encoder_cat:
                embedding_sizes[k] = v
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        kwargs.setdefault(
            "target_lags",
            {lag.feature_name: lag._state for lag in dataset.lags},
        )
        return cls(
            dataset.targets,
            encoder_cat=dataset.encoder_cat,
            # make sure the order is matched with that in `def encode`
            encoder_cont=dataset.encoder_cont + dataset.time_features + dataset.encoder_lag_features,
            decoder_cat=dataset.decoder_cat,
            # make sure the order is matched with that in `def decode` except target
            decoder_cont=dataset.decoder_cont + dataset.time_features + dataset.decoder_lag_features,
            embedding_sizes=embedding_sizes,
            x_categoricals=dataset.categoricals,
            **kwargs,
        )
