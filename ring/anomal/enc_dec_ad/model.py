import pytorch_forecasting
from ring.common.dataset import TimeSeriesDataset

pytorch_forecasting
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import torch
from ring.common.base_model import BaseAnormal
from ring.common.ml.rnn import get_rnn
from torch import nn

HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class EncoderDecoderAD(BaseAnormal):
    """Encoder-Decoder Architecture for Time-series Anomaly Detection"""

    def __init__(
        self,
        name: str = "enc_dec_ad",  # TODO:  this can be obtained by `self.__class__.__name__`
        cell_type: str = "LSTM",
        hidden_size: int = 5,
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        n_layers: int = 1,
        dropout: float = 0,
        output_size=1,
        x_categoricals: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        target_lags: Dict = {},
        mean: float = None,
        cov: float = None,
    ):
        # TODO: please complete the docstring
        """Encoder-Decoder Architecture for Time-series Anomaly Detection

        Args:
            name (str, optional): _description_. Defaults to "enc_dec_ad".
            cell_type (str, optional): _description_. Defaults to "LSTM".
            hidden_size (int, optional): _description_. Defaults to 5.
            embedding_sizes (Dict[str, Tuple[int, int]], optional): _description_. Defaults to {}.
            n_layers (int, optional): _description_. Defaults to 1.
            dropout (float, optional): _description_. Defaults to 0.
            output_size (int, optional): _description_. Defaults to 1.
            x_categoricals (List[str], optional): _description_. Defaults to [].
            encoder_cont (List[str], optional): _description_. Defaults to [].
            encoder_cat (List[str], optional): _description_. Defaults to [].
            target_lags (Dict, optional): _description_. Defaults to {}.
            mean (float, optional): _description_. Defaults to None.
            cov (float, optional): _description_. Defaults to None.
        """
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
