import torch
from torch import nn
from typing import List, Dict, Tuple
from copy import deepcopy

from ring.common.base_model import BaseModel
from ring.common.ml.rnn import get_rnn
from ring.common.ml.embeddings import AdditiveCategoricalEmbedding
from ring.common.dataset import TimeSeriesDataset


class RNNSeq2Seq(BaseModel):
    """
    A basic seq2seq model build around the basic lstm/gru
    """

    def __init__(
        self,
        targets: str,
        output_size: int,
        # hpyerparameters
        cell_type: str = "GRU",
        hidden_size: int = 16,
        n_layers: int = 1,
        dropout: float = 0.1,
        n_heads: int = 0,
        # data types
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        decoder_cont: List[str] = [],
        decoder_cat: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
    ):
        super().__init__()

        self._targets = targets
        self._encoder_cont = encoder_cont
        self._encoder_cat = encoder_cat
        self._decoder_cont = decoder_cont
        self._decoder_cat = decoder_cat
        self._n_heads = n_heads

        # 检查每个categorical的数据都存在对应的embedding_size
        assert all(
            [c in embedding_sizes for c in encoder_cat]
        ), "Every encoder categorical should have an embedding size"
        assert all(
            [c in embedding_sizes for c in decoder_cat]
        ), "Every decoder categorical should have an embedding size"
        assert (
            self._targets not in self._decoder_cont
        ), f"Target: {self._targets} should not in decoder_cont, which contains: {self._decoder_cont}"

        # 创建embedding
        self.categorical_embedding = AdditiveCategoricalEmbedding(
            hidden_size=hidden_size,
            embedding_sizes=embedding_sizes,
            encoder_cat=encoder_cat,
            decoder_cat=decoder_cat,
        )

        rnn_class = get_rnn(cell_type)
        rnn_kwargs = dict(
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        if n_layers > 1:
            rnn_kwargs.update(
                {
                    "dropout": dropout,
                }
            )

        # create encoder
        self.encoder = rnn_class(input_size=hidden_size, **rnn_kwargs)

        # create decoder
        if n_heads > 0:
            self.decoder = rnn_class(input_size=hidden_size + hidden_size, **rnn_kwargs)
        else:
            self.decoder = rnn_class(input_size=hidden_size, **rnn_kwargs)

        # create attention
        if n_heads > 0:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=n_heads,
                dropout=dropout,
                kdim=hidden_size,
                vdim=hidden_size,
                batch_first=True,
            )

        self.fc_out = nn.Linear(hidden_size, output_size)

    @property
    def target_positions(self):
        return [self._encoder_cont.index(tar) for tar in self._targets]

    def forward(self, x: Dict[str, torch.Tensor], **kwargs):
        encoder_cat = x["encoder_cat"]
        encoder_cont = x["encoder_cont"]
        decoder_cat = x["decoder_cat"]
        decoder_cont = x["decoder_cont"]
        batch_size = encoder_cont.size(0)
        decoder_sequence_length = decoder_cont.size(1) if decoder_cont.ndim > 0 else decoder_cat.size(1)

        encoder_cat_embedded, decoder_cat_embedded = self.categorical_embedding(encoder_cat, decoder_cat)

        # construct inputs
        encoder_input = encoder_cat_embedded
        decoder_input = decoder_cat_embedded

        encoder_outs, hidden_state = self.encoder(encoder_input)

        # multi head attention, using hidden_state as query, encoder's outs as key and value
        # TODO if we only using attention after lstm layer, it may make the code simpler
        if self._n_heads > 0:
            features = []
            for i in range(decoder_sequence_length):
                hidden_tensor = hidden_state if isinstance(hidden_state, torch.Tensor) else hidden_state[0]
                # query shape, [batch_size, 1, query_size], always taken the last layer output
                query = hidden_tensor[-1].reshape(batch_size, 1, -1)
                # key shape, [batch_size, encoder_sequence_length, hidden_size]
                attn_output = self.attention(query, encoder_outs, encoder_outs)[0].reshape((batch_size, -1))
                enhanced_decoder_input = torch.cat((decoder_input[:, i, :], attn_output), dim=1).unsqueeze(1)
                decoder_outs, hidden_state = self.decoder(enhanced_decoder_input, hidden_state)
                features.append(decoder_outs)
            features = torch.cat(features, dim=1)
        else:
            features, _ = self.decoder(decoder_input, hidden_state)

        return self.fc_out(features)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "RNNSeq2Seq":
        # update embedding sizes from kwargs
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = {}
        for k, v in dataset.embedding_sizes.items():
            if k in dataset.encoder_cat:
                embedding_sizes[k] = v
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)

        return cls(
            dataset.targets,
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont + dataset.time_features,
            decoder_cat=dataset.decoder_cat,
            decoder_cont=dataset.decoder_cont + dataset.time_features,
            embedding_sizes=embedding_sizes,
            **kwargs,
        )
