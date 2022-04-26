from types import CellType
from scipy.stats import multivariate_normal
from ring.common.dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from ring.common.base_model import BaseAnormal
from ring.common.ml.rnn import get_rnn
import numpy as np

HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class enc_dec_ad(BaseAnormal):
    def __init__(
        self,
        name: str = "enc_dec_ad",
        cell_type: str = "LSTM",
        hidden_size: int = 5,
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        n_layers: int = 1,
        dropout: float = 0,
        x_categoricals: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        target_lags: Dict = {},
        output_size: int = 1,
    ):
        super().__init__(
            name=name,
            embedding_sizes=embedding_sizes,
            target_lags=target_lags,
            x_categoricals=x_categoricals,
            encoder_cont=encoder_cont,
            encoder_cat=encoder_cat,
            cell_type=cell_type,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.hidden_size = hidden_size
        self.cell_type = cell_type

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

        enc_output, enc_hidden = self.encode(x)
        simulation = self.decode(x, hidden_state=enc_hidden)

        if mode == "predict":
            if self.cov is None or self.mean is None:
                raise ValueError("Need to fit first")

        return simulation

    def calculate_params(self, errors: List[np.ndarray]):
        """
        calculate specific post-training parameters in model
        """
        mean = np.mean(errors, axis=0)
        cov = np.cov(errors, rowvar=False, dtype=mean.dtype)
        if not cov.shape:
            cov = cov.reshape(1)

        return {"mean": mean, "cov": cov}

    def predict(self, output: tuple, **kwargs):
        """
        calculate specific post-predict outputs in model
        """

        self.mvnormal = multivariate_normal(kwargs["mean"], kwargs["cov"], allow_singular=True)
        score = -self.mvnormal.logpdf(output[0].reshape(-1, len(self._encoder_cont)).data.cpu().numpy())
        return score, output[1]
