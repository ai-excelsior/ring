import torch
from torch import nn
from typing import List, Dict, Tuple
from copy import deepcopy

# from ring.common.ml.rnn import get_rnn
# from ring.common.ml.embeddings import MultiEmbedding
from ring.common.base_model import AutoRegressiveBaseModelWithCovariates
from ring.common.dataset import TimeSeriesDataset


class ReccurentNetwork(AutoRegressiveBaseModelWithCovariates):
    """
    A basic seq2seq model build around the basic lstm/gru
    """

    def forward(self, x: Dict[str, torch.Tensor]):
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
        embedding_sizes = deepcopy(dataset.embedding_sizes)
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        lags = {name: lag for name, lag in dataset.lags.items() if name in dataset.targets}
        kwargs.setdefault(
            "target_lags",
            {name: {f"{name}_lagged_by_{lag}": lag for lag in lags.get(name, [])} for name in lags},
        )
        return cls(
            dataset.targets,
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont,
            decoder_cat=dataset.decoder_cat,
            decoder_cont=dataset.decoder_cont,
            embedding_sizes=embedding_sizes,
            **kwargs,
        )
