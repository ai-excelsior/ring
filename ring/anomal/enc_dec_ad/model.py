import pytorch_forecasting
from ring.common.dataset import TimeSeriesDataset

pytorch_forecasting
import torch
from torch import nn
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from ring.common.base_model import BaseAnormal
from ring.common.ml.rnn import get_rnn

HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class enc_dec_ad(BaseAnormal):
    def __init__(
        self,
        name: str = "enc_dec_ad",
        cell_type: str = "LSTM",
        hidden_size: int = 5,
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        n_layers: tuple = (1, 1),
        dropout: tuple = (0, 0),
        output_size=1,
        x_categoricals: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        target_lags: Dict = {},
        mean: float = None,
        cov: float = None,
    ):
        super().__init__(
            name=name,
            embedding_sizes=embedding_sizes,
            target_lags=target_lags,
            x_categoricals=x_categoricals,
            encoder_cont=encoder_cont,
            encoder_cat=encoder_cat,
        )

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.cell_type = cell_type

        self.mean = mean
        self.cov = cov

        # only cont are considered, so `input_size` is this way
        self.encoder = get_rnn(cell_type)(
            input_size=len(self._encoder_cont),
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            bias=True,
            dropout=self.dropout,
        )

        self.decoder = get_rnn(cell_type)(
            input_size=len(self._encoder_cont),
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            bias=True,
            dropout=self.dropout,
        )

        self.output_projector_decoder = nn.Linear(self.hidden_size, len(self._encoder_cont))

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs):
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = deepcopy(dataset.embedding_sizes)
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        return cls(
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont,
            embedding_sizes=embedding_sizes,
            x_categoricals=dataset.categoricals,
            **kwargs
        )

    def forward(self, x: Dict[str, torch.Tensor], mode=None, **kwargs) -> Dict[str, torch.Tensor]:

        enc_hidden = self.encode(x)
        simulation = self.decode(x, hidden_state=enc_hidden)

        if mode == "predict":
            if self.cov is None or self.mean is None:
                raise ValueError("Need to fit first")

        return simulation
