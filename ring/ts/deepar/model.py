import torch
from torch import nn
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from ring.common.base_model import AutoRegressiveBaseModelWithCovariates
from ring.common.dataset import TimeSeriesDataset

HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class DeepAR(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        targets: str,
        output_size: int,
        # hpyerparameters
        cell_type: str = "GRU",
        hidden_size: int = 16,
        n_layers: int = 1,
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
            nn.Linear(hidden_size, 2)
            if len(self._targets) == 1
            else [nn.Linear(hidden_size, 2) for _ in range(len(self._targets))]
        )

    def decode(
        self,
        x: Dict[str, torch.Tensor],
        hidden_state: HIDDENSTATE,
        first_target: torch.Tensor,
        n_samples: int = None,
        **_,
    ) -> torch.Tensor:

        self._phase = "decode"
        decoder_cat, decoder_cont = x["decoder_cat"], torch.cat(
            [x["decoder_target"], x["decoder_cont"]], dim=-1
        )
        input_vector = self.construct_input_vector(decoder_cat, decoder_cont, first_target)
        if n_samples is None:  # the training mode where the target values are actually known
            output, _ = self._decode(
                input_vector=input_vector, hidden_state=hidden_state, lengths=x["decoder_length"]
            )
        else:  # the prediction mode, in which the target values are unknown
            input_vector = input_vector.repeat_interleave(n_samples, 0)
            hidden_state = self.decoder.repeat_interleave(hidden_state, n_samples)
            output = self.decode_autoregressive(
                self._decode,
                input_vector=input_vector,
                hidden_state=hidden_state,
                n_decoder_steps=x["decoder_length"][0],
            )
        return output

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None, mode=None):
        hidden_state = self.encode(x)
        first_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].shape[0], device=x["encoder_cont"].device),
            x["encoder_length"] - 1,
            self.target_positions.unsqueeze(-1),
        ].t()

        # if mode == "predict":
        #     return self.decode(x, hidden_state=hidden_state, first_target=first_target, n_samples=100)

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"

        return self.decode(x, hidden_state=hidden_state, first_target=first_target)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "DeepAR":
        # update embedding sizes from kwargs
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = {}
        for k, v in dataset.embedding_sizes.items():
            if k in dataset.encoder_cat:
                embedding_sizes[k] = v
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        lags = {name: lag for name, lag in dataset.lags.items() if name in dataset.targets}
        kwargs.setdefault(
            "target_lags",
            {name: {f"{name}_lagged_by_{lag}": lag for lag in lags.get(name, [])} for name in lags},
        )
        return cls(
            targets=dataset.targets,
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont,
            decoder_cat=dataset.decoder_cat,
            decoder_cont=dataset.decoder_cont,
            embedding_sizes=embedding_sizes,
            x_categoricals=dataset.categoricals,
            **kwargs,
        )
