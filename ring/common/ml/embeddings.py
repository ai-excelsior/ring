from functools import cached_property
import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Union
from ..utils import get_default_embedding_size


class TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return super().forward(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class StackCategoricalEmbedding(nn.Module):
    """An embedding layer to embed categoricals with stack manner"""

    def __init__(
        self,
        embedding_sizes: Dict[str, Tuple[int, int]],
        encoder_cat: List[str] = [],
        decoder_cat: List[str] = [],
        ignore_keys: List[str] = [],
    ) -> None:
        super().__init__()
        self._encoder_cat = encoder_cat
        self._decoder_cat = decoder_cat
        self._empty_tensor = torch.tensor([])

        embedding_keys = [
            key
            for key in [*encoder_cat, *decoder_cat]
            if (key in embedding_sizes) and (key not in ignore_keys)
        ]
        self.embeddings = nn.ModuleDict(
            {key: nn.Embedding(embedding_sizes[key][0], embedding_sizes[key][1]) for key in embedding_keys}
        )

    @property
    def embedding_sizes(self) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: (encoder_embedding_size, decoder_embedding_size)
        """
        return (
            sum([self.embeddings[key].embedding_dim for key in self._encoder_cat if key in self.embeddings]),
            sum([self.embeddings[key].embedding_dim for key in self._decoder_cat if key in self.embeddings]),
        )

    def forward(self, encoder_cat: torch.Tensor, decoder_cat: torch.Tensor):
        """
        Args:
            encoder_cat (torch.Tensor): [batch_size, sequence_length, n_encoder_cat]
            decoder_cat (torch.Tensor): [batch_size, sequence_length, n_decoder_cat]
        """
        encoder_embedding_size, decoder_embedding_size = self.embedding_sizes

        if encoder_embedding_size == 0:
            encoder_embedded = torch.zeros(
                (*encoder_cat.shape[:-1], encoder_embedding_size), device=encoder_cat.device
            )
        else:
            encoder_embedded = torch.cat(
                [
                    self.embeddings[key](encoder_cat[..., i])
                    for i, key in enumerate(self._encoder_cat)
                    if key in self.embeddings
                ],
                dim=-1,
            )

        if decoder_embedding_size == 0:
            decoder_embedded = torch.zeros(
                (*decoder_cat.shape[:-1], decoder_embedding_size), device=decoder_cat.device
            )
        else:
            decoder_embedded = torch.cat(
                [
                    self.embeddings[key](decoder_cat[..., i])
                    for i, key in enumerate(self._decoder_cat)
                    if key in self.embeddings
                ],
                dim=-1,
            )
        return (encoder_embedded, decoder_embedded)


class AdditiveCategoricalEmbedding(nn.Module):
    """embedding categoricals with additive manner"""

    def __init__(
        self,
        hidden_size: int,
        embedding_sizes: Dict[str, Tuple[int, int]],
        encoder_cat: List[str] = [],
        decoder_cat: List[str] = [],
        ignore_keys: List[str] = [],
    ) -> None:
        super().__init__()

        self._encoder_cat = encoder_cat
        self._decoder_cat = decoder_cat
        self._hidden_size = hidden_size

        embedding_keys = [
            key
            for key in [*encoder_cat, *decoder_cat]
            if (key in embedding_sizes) and (key not in ignore_keys)
        ]
        self.embeddings = nn.ModuleDict(
            {key: nn.Embedding(embedding_sizes[key][0], hidden_size) for key in embedding_keys}
        )

    @property
    def embedding_sizes(self) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: (encoder_embedding_size, decoder_embedding_size)
        """
        return (
            sum([self.embeddings[key].embedding_dim for key in self._encoder_cat if key in self.embeddings]),
            sum([self.embeddings[key].embedding_dim for key in self._decoder_cat if key in self.embeddings]),
        )

    def forward(self, encoder_cat: torch.Tensor, decoder_cat: torch.Tensor):
        """
        Args:
            encoder_cat (torch.Tensor): [batch_size, sequence_length, n_encoder_cat]
            decoder_cat (torch.Tensor): [batch_size, sequence_length, n_decoder_cat]
        """
        encoder_embedding_size, decoder_embedding_size = self.embedding_sizes
        if encoder_embedding_size == 0:
            encoder_embedded = torch.zeros(
                (*encoder_cat.shape[:-1], self._hidden_size), device=encoder_cat.device
            )
        else:
            encoder_embedded = torch.sum(
                torch.stack(
                    [
                        self.embeddings[key](encoder_cat[..., i])
                        for i, key in enumerate(self._encoder_cat)
                        if key in self.embeddings
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

        if decoder_embedding_size == 0:
            decoder_embedded = torch.zeros(
                (*decoder_cat.shape[:-1], self._hidden_size), device=decoder_cat.device
            )
        else:
            decoder_embedded = torch.sum(
                torch.stack(
                    [
                        self.embeddings[key](decoder_cat[..., i])
                        for i, key in enumerate(self._decoder_cat)
                        if key in self.embeddings
                    ],
                    dim=-1,
                ),
                dim=-1,
            )
        return (encoder_embedded, decoder_embedded)


class UnaryConv1d(nn.Module):
    """Simulate Conv1d, but operate on an empty tensor, only perpose is to make code simpler"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert in_channels == 0
        self._padding = padding
        self._dilation = dilation
        self._kernel_size = kernel_size
        self._stride = stride

    def get_l_out(self, l_in: int) -> int:
        return (l_in + 2 * self._padding - self._dilation * (self._kernel_size - 1) - 1) // self._stride + 1

    def forward(self, x: torch.Tensor, *args, **kwarg) -> torch.Tensor:
        assert x.numel() == 0, "only support empty tensor"

        if x.ndim == 3:
            return torch.zeros((x.shape[0], 1, self.get_l_out(x.shape[2]))).to(x.device)
        else:
            return torch.zeros((1, self.get_l_out(x.shape[1]))).to(x.device)


class ConvEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        encoder_cont: List[str],
        decoder_cont: List[str],
    ) -> None:
        super().__init__()

        self._encoder_cont = encoder_cont
        self._decoder_cont = decoder_cont

        # vars after processed
        inter_cont = self.inter_cont
        encoder_cont = self.encoder_cont
        decoder_cont = self.decoder_cont

        self._indices_of_encoder_inter = torch.tensor(
            [key in inter_cont for key in self._encoder_cont], dtype=torch.bool
        )
        self._indices_of_decoder_inter = torch.tensor(
            [key in inter_cont for key in self._decoder_cont], dtype=torch.bool
        )
        self._indices_of_encoder = torch.tensor(
            [key in encoder_cont for key in self._encoder_cont], dtype=torch.bool
        )
        self._indices_of_decoder = torch.tensor(
            [key in decoder_cont for key in self._decoder_cont], dtype=torch.bool
        )

        # create conv layers
        get_conv = lambda x: nn.Conv1d if x > 0 else UnaryConv1d
        self.inter_conv = get_conv(len(inter_cont))(
            in_channels=len(inter_cont),
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.encoder_cov = get_conv(len(encoder_cont))(
            in_channels=len(encoder_cont),
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.decoder_cov = get_conv(len(decoder_cont))(
            in_channels=len(decoder_cont),
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

    @cached_property
    def inter_cont(self) -> List[str]:
        """intersection of encoder_cont and decoder_cont"""
        return [key for key in self._encoder_cont if key in self._decoder_cont]

    @cached_property
    def encoder_cont(self) -> List[str]:
        """The features only in encoder cont"""
        return [key for key in self._encoder_cont if key not in self.inter_cont]

    @cached_property
    def decoder_cont(self) -> List[str]:
        """The features only in decoder cont"""
        return [key for key in self._decoder_cont if key not in self.inter_cont]

    def forward(
        self, encoder_cont: torch.Tensor, decoder_cont: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out: torch.Tensor = self.inter_conv(
            encoder_cont[..., self._indices_of_encoder_inter].transpose(-1, -2)
        ) + self.encoder_cov(encoder_cont[..., self._indices_of_encoder].transpose(-1, -2))

        decoder_out: torch.Tensor = self.inter_conv(
            decoder_cont[..., self._indices_of_decoder_inter].transpose(-1, -2)
        ) + self.decoder_cov(decoder_cont[..., self._indices_of_decoder].transpose(-1, -2))

        return encoder_out.transpose(-1, -2), decoder_out.transpose(-1, -2)


TIME_EMBEDDING_SIZES = {
    "month": (12, get_default_embedding_size(12)),  # 1 encode
    "day": (31, get_default_embedding_size(31)),  # 1 encode
    "hour": (24, get_default_embedding_size(24)),  # 0 encode
    "minute": (60, get_default_embedding_size(60)),  # 0 encode
    "second": (60, get_default_embedding_size(60)),  # 0 encode
    "day_of_week": (7, get_default_embedding_size(7)),  # 0 encode
    "day_of_year": (366, get_default_embedding_size(366)),  # 1 encode
}


class TimeEmbedding(nn.Module):
    """An embedding layer to embed time components"""

    def __init__(
        self,
        hidden_size: int,
        encoder_cat: List[str] = [],
        decoder_cat: List[str] = [],
    ) -> None:
        super().__init__()
        embedding_sizes = TIME_EMBEDDING_SIZES

        self._encoder_cat = encoder_cat
        self._decoder_cat = decoder_cat
        self._hidden_size = hidden_size

        self._embedding_keys = [key for key in [*encoder_cat, *decoder_cat] if key in embedding_sizes]
        self.embeddings = nn.ModuleDict(
            {key: nn.Embedding(embedding_sizes[key][0], hidden_size) for key in self._embedding_keys}
        )

    @property
    def embedding_keys(self) -> List[str]:
        return self._embedding_keys

    @cached_property
    def encoder_cat(self) -> List[str]:
        """The features only in encoder cont"""
        return [key for key in self._encoder_cat if key in self._embedding_keys]

    @cached_property
    def decoder_cat(self) -> List[str]:
        """The features only in decoder cont"""
        return [key for key in self._decoder_cat if key in self._embedding_keys]

    def forward(self, encoder_cat: torch.Tensor, decoder_cat: torch.Tensor):
        """
        Args:
            encoder_cat (torch.Tensor): [batch_size, sequence_length, n_encoder_cat]
            decoder_cat (torch.Tensor): [batch_size, sequence_length, n_decoder_cat]
        """
        if len(self.encoder_cat) == 0:
            encoder_embedded = torch.zeros(
                (*encoder_cat.shape[:-1], self._hidden_size), device=encoder_cat.device
            )
        else:
            encoder_embedded = torch.sum(
                torch.stack(
                    [
                        self.embeddings[key](encoder_cat[..., i])
                        for i, key in enumerate(self._encoder_cat)
                        if key in self.embeddings
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

        if len(self.decoder_cat) == 0:
            decoder_embedded = torch.zeros(
                (*decoder_cat.shape[:-1], self._hidden_size), device=decoder_cat.device
            )
        else:
            decoder_embedded = torch.sum(
                torch.stack(
                    [
                        self.embeddings[key](decoder_cat[..., i])
                        for i, key in enumerate(self._decoder_cat)
                        if key in self.embeddings
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

        return (encoder_embedded, decoder_embedded)


class MLPEmbedding(nn.Module):
    """embedding encoder_cont encoder_cat decoder_cont decoder_cat with one layer mlp"""

    def __init__(
        self,
        hidden_size: int,
        embedding_sizes: Dict[str, Tuple[int, int]],
        encoder_cat: List[str] = [],
        decoder_cat: List[str] = [],
        encoder_cont: List[str] = [],
        decoder_cont: List[str] = [],
    ) -> None:
        super().__init__()

        self.time_embedding = TimeEmbedding(
            hidden_size=hidden_size,
            encoder_cat=encoder_cat,
            decoder_cat=decoder_cat,
        )
        self.categorical_embedding = StackCategoricalEmbedding(
            embedding_sizes=embedding_sizes,
            encoder_cat=encoder_cat,
            decoder_cat=decoder_cat,
            ignore_keys=self.time_embedding.embedding_keys,
        )

        encoder_embedding_size, decoder_embedding_size = self.categorical_embedding.embedding_sizes
        self.encoder_fc = nn.Linear(
            in_features=len(encoder_cont) + encoder_embedding_size,
            out_features=hidden_size,
        )
        self.decoder_fc = nn.Linear(
            in_features=len(decoder_cont) + decoder_embedding_size,
            out_features=hidden_size,
        )

    def forward(
        self,
        encoder_cont: torch.Tensor,
        decoder_cont: torch.Tensor,
        encoder_cat: torch.Tensor,
        decoder_cat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_time_embedded, decoder_time_embedded = self.time_embedding(encoder_cat, decoder_cat)
        encoder_cat_embedded, decoder_cat_embedded = self.categorical_embedding(encoder_cat, decoder_cat)
        encoder_x = torch.cat((encoder_cont, encoder_cat_embedded), dim=-1)
        decoder_x = torch.cat((decoder_cont, decoder_cat_embedded), dim=-1)
        encoder_out = self.encoder_fc(encoder_x)
        decoder_out = self.decoder_fc(decoder_x)
        return encoder_out + encoder_time_embedded, decoder_out + decoder_time_embedded


class TrigonometricPositionalEmbedding(nn.Module):
    """三角函数位置编码"""

    def __init__(self, sequence_length: int, hidden_size: int):
        super().__init__()

        # [1, sequence_length, hidden_size]
        self.positions = torch.zeros((sequence_length, hidden_size))
        X = torch.arange(sequence_length, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size
        )
        self.positions[:, 0::2] = torch.sin(X)
        self.positions[:, 1::2] = torch.cos(X)

    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            time_idx (torch.Tensor): [batch_size, sequence_length]

        Returns:
            Tensor: a positon embedded tensor, [batch_size, sequence_length, hidden_size]
        """
        return self.positions[time_idx].to(time_idx.device)


class MultiEmbedding(nn.Module):
    def __init__(
        self,
        embedding_sizes: Dict[str, Tuple[int, int]],
        categorical_groups: Dict[str, List[str]],
        embedding_paddings: List[str],
        x_categoricals: List[str],
        max_embedding_size: int = None,
    ):
        super().__init__()
        self.embedding_sizes = embedding_sizes
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals
        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            # convert to list to become mutable
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            self.embedding_sizes[name][1] = embedding_size
            if name in self.categorical_groups:  # embedding bag if related embeddings
                self.embeddings[name] = TimeDistributedEmbeddingBag(
                    self.embedding_sizes[name][0], embedding_size, mode="sum", batch_first=True
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

    def total_embedding_size(self) -> int:
        return sum([size[1] for size in self.embedding_sizes.values()])

    def names(self) -> List[str]:
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self) -> List[str]:
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str) -> Union[nn.Embedding, TimeDistributedEmbeddingBag]:
        return self.embeddings[name]

    def forward(self, x: torch.Tensor, flat: bool = False) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        out = {}
        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                out[name] = emb(
                    x[
                        ...,
                        [self.x_categoricals.index(cat_name) for cat_name in self.categorical_groups[name]],
                    ]
                )
            else:
                out[name] = emb(x[..., self.x_categoricals.index(name)])
        if flat:
            out = torch.cat([v for v in out.values()], dim=-1)
        return out
