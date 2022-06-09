from turtle import forward
import torch
from torch import nn
from typing import List, Dict, Tuple
from copy import deepcopy
from ring.common.base_model import BaseLong
from ring.common.ml.embeddings import DataEmbedding
from ring.common.dataset import TimeSeriesDataset


class Informer(BaseLong):
    def __init__(
        self,
        targets: List[str] = [],
        output_size: int = 1,
        context_length: int = 2,
        prediction_length: int = 1,
        token_length: int = 1,
        # hpyerparameters
        n_heads: int = 2,
        hidden_size: int = 64,
        fcn_size: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.1,
        # data types
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        decoder_cont: List[str] = [],
        decoder_cat: List[str] = [],
        target_lags: Dict = {},
        freq: str = "h",
    ):

        super().__init__(
            targets=targets,
            context_length=context_length,
            prediction_length=prediction_length,
            token_length=token_length,
            fcn_size=fcn_size,
            n_heads=n_heads,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            encoder_cont=encoder_cont,
            encoder_cat=encoder_cat,
            decoder_cont=decoder_cont,
            decoder_cat=decoder_cat,
            target_lags=target_lags,
            output_size=output_size,
            freq=freq,
        )

    def forward(self, x: Dict[str, torch.Tensor], **kwargs):
        enc_out, _ = self.encode(x)
        prediction = self.decode(x, enc_out)

        return self.projection(prediction)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "Informer":
        """

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to ``__init__`` method.

        Returns:
            Informer
        """

        return cls(
            dataset.targets,
            output_size=len(dataset.targets),
            context_length=dataset.get_parameters().get("indexer").get("params").get("look_back"),
            prediction_length=dataset.get_parameters().get("indexer").get("params").get("look_forward"),
            token_length=dataset.get_parameters().get("indexer").get("params").get("look_forward") // 2,
            encoder_cont=dataset.encoder_cont,
            encoder_cat=dataset.encoder_cat,
            decoder_cont=dataset.decoder_cont,
            decoder_cat=dataset.decoder_cat,
            freq=dataset._freq,
            **kwargs
        )
