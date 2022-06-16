from scipy.stats import multivariate_normal
from ring.common.dataset import TimeSeriesDataset
import torch
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from ring.common.base_model import BaseAnormal
import numpy as np

HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class EncoderDecoderAD(BaseAnormal):
    def __init__(
        self,
        name: str = "enc_dec_ad",
        cell_type: str = "LSTM",
        hidden_size: int = 5,
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        n_layers: int = 1,
        dropout: float = 0,
        targets: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        target_lags: Dict = {},
        output_size: int = 1,
        return_enc: bool = False,
        encoderdecodertype: str = "RNN",
    ):
        super().__init__(
            name=name,
            embedding_sizes=embedding_sizes,
            target_lags=target_lags,
            targets=targets,
            encoder_cont=encoder_cont,
            encoder_cat=encoder_cat,
            cell_type=cell_type,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            return_enc=return_enc,
            encoderdecodertype=encoderdecodertype,
        )

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs):
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = {}
        for k, v in dataset.embedding_sizes.items():
            if k in dataset.encoder_cat:
                embedding_sizes[k] = v
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        return cls(
            targets=dataset.targets,
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont,
            embedding_sizes=embedding_sizes,
            **kwargs
        )

    def forward(self, x: Dict[str, torch.Tensor], mode=None, **kwargs) -> Dict[str, torch.Tensor]:
        print("begin predict")
        simulation = self.encoderdecoder(x)
        # only consider recon of `cont`
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
        score = -self.mvnormal.logpdf(output[0].reshape(-1, len(self._encoder_cont)).cpu())

        return score.reshape(output[1].shape[0], -1), output[1]
