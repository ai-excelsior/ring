from types import CellType
from scipy.stats import multivariate_normal
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
        x_categoricals: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        target_lags: Dict = {},
        output_size: int = 1,
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
        enc_hidden = self.encode(x)
        simulation = self.decode(x, hidden_state=enc_hidden)

        return simulation

    def calculate_params(self, error_vectors: List[np.ndarray]):
        """
        calculate specific post-training parameters in model
        """
        mean = np.mean(error_vectors, axis=0)
        cov = np.cov(error_vectors, rowvar=False, dtype=mean.dtype)
        if not cov.shape:
            cov = cov.reshape(1)

        return {"mean": mean, "cov": cov}

    def predict(self, output: tuple, **kwargs):
        """
        calculate specific post-predict outputs in model
        """

        self.mvnormal = multivariate_normal(kwargs["mean"], kwargs["cov"], allow_singular=True)
        score = -self.mvnormal.logpdf(output[0].reshape(-1, len(self._encoder_cont)))

        return score.reshape(1, -1), output[1]
