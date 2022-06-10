from torch import nn
import torch
from .dataset import TimeSeriesDataset
from functools import cached_property
from typing import Callable, Dict, List, Tuple, Union
from ring.common.ml.embeddings import MultiEmbedding, DataEmbedding
from ring.common.ml.utils import to_list
from ring.common.ml.rnn import get_rnn
from ring.common.base_en_decoder import (
    AutoencoderType,
    RnnType,
    VariAutoencoderType,
    ConvLayer,
    Decoder,
    DecoderLayer,
    EncoderLayer,
    Encoder,
)
from ring.common.attention import AttentionLayer, ProbAttention, FullAttention

HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseModel(nn.Module):
    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "BaseModel":

        raise NotImplementedError()


class AutoRegressiveBaseModelWithCovariates(BaseModel):
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
        super().__init__()

        self._targets = targets
        self._encoder_cont = encoder_cont
        self._encoder_cat = encoder_cat
        self._decoder_cont = decoder_cont
        self._decoder_cat = decoder_cat
        self._x_categoricals = x_categoricals
        self._targets_lags = target_lags
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
        self.encoder_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=[],
            categorical_groups={},
            x_categoricals=x_categoricals,
        )

        self.decoder_embeddings = (
            self.encoder_embeddings
            if self._decoder_cat == self._encoder_cat
            else MultiEmbedding(
                embedding_sizes={k: v for k, v in embedding_sizes.items() if k in self._decoder_cat},
                embedding_paddings=[],
                categorical_groups={},
                x_categoricals=[x for x in x_categoricals if x in self._decoder_cat],
            )
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
        self._phase = "encode"
        self.encoder = rnn_class(input_size=self._encoder_input_size, **rnn_kwargs)
        self.decoder = (
            self.encoder
            if not (self.has_time_varying_unknown_cont or self.has_time_varying_unknown_cat)
            else rnn_class(input_size=self._decoder_input_size, **rnn_kwargs)
        )

    @property
    def categoricals_embedding(self) -> MultiEmbedding:
        return (
            getattr(self, "encoder_embeddings", None)
            if self._phase == "encode"
            else getattr(self, "decoder_embeddings", None)
        )

    @cached_property
    def _encoder_input_size(self) -> int:
        """the actual input size/dim of the encoder after the categorical embedding"""
        return len(self._encoder_cont) + self.encoder_embeddings.total_embedding_size()

    @cached_property
    def _decoder_input_size(self) -> int:
        """the actual input size/dim of the decoder after the categorical embedding"""
        return len(self._decoder_cont) + self.decoder_embeddings.total_embedding_size()

    @property
    def reals(self) -> List[str]:
        """lists of reals in the encoder or decoder sequence"""
        return self._encoder_cont if self._phase == "encode" else self._decoder_cont

    @property
    def reals_indices(self) -> List[int]:
        """lists of the indices of reals in `x["encoder_cont"]` or `x["decoder_cont"]`"""
        return [self._encoder_cont.index(name) for name in self.reals if name != "_ZERO_"]

    @property
    def categoricals(self) -> List[str]:
        """lists of categoricals in the encoder or decoder sequence"""
        return self._encoder_cat if self._phase == "encode" else self._decoder_cat

    @property
    def categoricals_indices(self) -> List[int]:
        """lists of the indices of categoricals in `x["encoder_cat"]` or `x["decoder_cat"]`"""
        return [self._x_categoricals.index(name) for name in self.categoricals]

    @property
    def target_positions(self) -> torch.LongTensor:
        """Target positions in the encoder or decoder tensor

        Note that when `time_varying_unknown_reals` is present, `target_positions` gives the indices
        of target columns after dropping `time_varying_unknown_reals` from `x["decoder_cont"]`

        Returns:
            torch.LongTensor: tensor of positions.
        """
        pos = torch.tensor(
            [self._encoder_cont.index(name) for name in to_list(self._targets)],
            dtype=torch.long,
        )
        a = torch.tensor(
            [i for i, p in enumerate(self.reals_indices) if p in pos],
            dtype=torch.long,
        )
        # device=self.device,
        return a

    @property
    def lagged_target_positions(self) -> Dict[int, torch.LongTensor]:
        """Lagged target positions in the encoder or decoder tensor

        Note that when `time_varying_unknown_reals` is present, `lagged_target_positions` gives the indices
        of lagged target columns after dropping `time_varying_unknown_reals` from `x["decoder_cont"]`

        Returns:
            Dict[int, torch.LongTensor]: dictionary mapping integer lags to tensor of variable positions.
        """
        # todo: expand for categorical targets
        if len(self._targets_lags) == 0:
            pos = {}
        else:
            # extract lags which are the same across all targets
            lags = list(next(iter(self._targets_lags.values())).values())
            lag_names = {lag_name: [] for lag_name in lags}
            for targeti_lags in self._targets_lags.values():
                for name, l in targeti_lags.items():
                    lag_names[l].append(name)

            pos = {
                lag: torch.tensor(
                    [self._encoder_cont.index(name) for name in to_list(names)],
                    dtype=torch.long,
                    device=self._encoder_cont.device,
                )
                for lag, names in lag_names.items()
            }

        return {
            k: (torch.tensor(self.reals_indices, device=self.reals_indices.device) == v).nonzero(
                as_tuple=True
            )[0]
            if len(v) == 1
            else torch.stack(
                [
                    (torch.tensor(self.reals_indices, device=self.reals_indices.device) == item).nonzero(
                        as_tuple=True
                    )[0]
                    for item in v
                ]
            ).nonzero(as_tuple=True)[0]
            for k, v in pos.items()
        }

    @cached_property
    def has_time_varying_unknown_cont(self) -> bool:
        return set(self._encoder_cont) != set(self._decoder_cont)

    @cached_property
    def has_time_varying_unknown_cat(self) -> bool:
        return set(self._encoder_cat) != set(self._decoder_cat)

    def construct_input_vector(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        first_target: torch.Tensor = None,
    ) -> torch.Tensor:
        # create input vector
        input_vector = []
        # NOTE: the real-valued variables always come first in the input vector
        if len(self.reals) > 0:
            input_vector.append(x_cont[..., self.reals_indices].clone())

        if len(self.categoricals) > 0:
            embeddings = self.categoricals_embedding(x_cat[..., self.categoricals_indices], flat=True)
            input_vector.append(embeddings)

        input_vector = torch.cat(input_vector, dim=-1)
        # shift the target variables by one time step into the future
        # when `encode`, this make sure the non-overlapping of `hidden_state` and `input_vector` used in `decode` lately
        # when `decode`, this make sure the first predict target is known thus can be directly taken from `input_vector`
        input_vector[..., self.target_positions.to(input_vector.device)] = input_vector[
            ..., self.target_positions.to(input_vector.device)
        ].roll(shifts=1, dims=1)

        if first_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions.to(input_vector.device)] = first_target
        # else:  # or drop the last time step, can be done by x["encoder_length"] - 1
        #     input_vector = input_vector[:, :-1]
        return input_vector

    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, HIDDENSTATE]:
        """Encode a sequence into hidden state and make backcasting predictions

        Args:
            x (Dict[str, torch.Tensor]): the input dictionary

        Returns:
            Tuple[torch.Tensor, HiddenState]:
                * the prediction on the encoding sequence
                * the last hidden state
        """
        self._phase = "encode"
        encoder_cat, encoder_cont, lengths = x["encoder_cat"], x["encoder_cont"], x["encoder_length"] - 1
        assert lengths.min() > 0
        input_vector = self.construct_input_vector(encoder_cat, encoder_cont)
        _, hidden_state = self.encoder(input_vector, lengths=lengths, enforce_sorted=False)
        return hidden_state

    def decode_autoregressive(
        self,
        decode_one_step: Callable,
        input_vector: torch.Tensor,
        hidden_state: HIDDENSTATE,
        n_decoder_steps: int,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:

        # the autoregression loop
        predictions = list()
        # the first predicted target can be directly obtained from data
        # because the last row of `encoder_cont`(known) has been taken to replace the first roll of rolled `input_vector`
        normalized_target = [input_vector[:, 0, self.target_positions.to(input_vector.device)]]
        # the autoregression loop
        for idx in range(n_decoder_steps):
            _input_vector = input_vector[:, [idx]]
            # take the last predicted target values as the input for the current prediction step
            _input_vector[:, 0, self.target_positions.to(_input_vector.device)] = normalized_target[-1]
            for lag, lag_positions in self.lagged_target_positions.items():
                # lagged values are depleted: if the current prediction step is beyond the lag
                if idx > lag and len(lag_positions) > 0:
                    _input_vector[:, 0, lag_positions] = normalized_target[-lag]

            output_, hidden_state = decode_one_step(
                input_vector=_input_vector,
                hidden_state=hidden_state,
                **kwargs,
            )
            output = [o.squeeze(1) for o in output_] if isinstance(output_, list) else output_.squeeze(1)
            predictions.append(output)
            normalized_target.append(output)

        return torch.stack(predictions, dim=1)

    def decode(
        self,
        x: Dict[str, torch.Tensor],
        hidden_state: HIDDENSTATE,
        first_target: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        """Decode a hidden state and an input sequence into forcasting and backcasting predictions.

        Args:
            x (Dict[str, torch.Tensor]): the input dictionary
            hidden_state (HiddenState): the last hidden state from the encoder
            first_target (torch.Tensor): the target value used in the first time step of the input sequence,
                which should be the target values of the last time step of the encoding sequence.

        Returns:
            torch.Tensor: the prediction on the decoding sequence
        """
        self._phase = "decode"
        decoder_cat, decoder_cont = x["decoder_cat"], x["decoder_cont"]
        input_vector = self.construct_input_vector(decoder_cat, decoder_cont, first_target)
        if self.training:  # the training mode where the target values are actually known
            output, _ = self._decode(
                input_vector=input_vector, hidden_state=hidden_state, lengths=x["decoder_length"]
            )
        else:  # the prediction mode, in which the target values are unknown
            output = self.decode_autoregressive(
                self._decode,
                input_vector=input_vector,
                hidden_state=hidden_state,
                # TODO: this does not work with randomized decoder length
                n_decoder_steps=x["decoder_length"][0],
            )
        return output

    def _decode(
        self,
        input_vector: torch.Tensor,
        hidden_state: HIDDENSTATE,
        lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, HIDDENSTATE]:
        """The actual decoding function for single or multiple time steps

        Args:
            input_vector (torch.Tensor): in the prediction mode, `x[..., self.target_positions]` should be
                already filled in with the predicted target values obtained in previous steps
                (e.g., performed in `self.decode_autoregressive`)
            hidden_state (HiddenState): RNN's hidden state from either the previous time step or the encoder
            lengths (torch.Tensor, optional): the input sequence length. Defaults to None.

        Returns:
            TODO: double check this docstring
            Tuple[torch.Tensor, HiddenState]:
                * the raw output, which could be either the normalized prediction values or the prediction
                    parameter, e.g., \mu and \sigma of the prediction distribution in the classification case
                * the hidden state at the last time step
        """
        output, hidden_state_ = self.decoder(
            input_vector, hidden_state, lengths=lengths, enforce_sorted=False
        )
        output = (
            self.output_projector_decoder(output)  # single target
            if len(self._targets) == 1
            else torch.concat([projector(output) for projector in self.output_projector_decoder], dim=-1)
        )
        return output, hidden_state_


class BaseAnormal(BaseModel):
    def __init__(
        self,
        name: str = None,
        # hpyerparameters
        # data types
        encoderdecodertype: str = "RNN",
        targets: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        target_lags: Dict = {},
        cell_type: str = "LSTM",
        hidden_size: int = 8,
        n_layers: int = 1,
        dropout: float = 0,
        return_enc: bool = False,
        steps: int = 1,
    ):
        super().__init__()
        self._targets = targets
        self._encoder_cont = encoder_cont
        self._encoder_cat = encoder_cat
        self._targets_lags = target_lags

        self.encoder_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=[],
            categorical_groups={},
            x_categoricals=encoder_cat,
        )
        # encoder-decoder submodule
        if encoderdecodertype == "RNN":
            self.encoderdecoder = RnnType(
                cont_size=len(self._encoder_cont),
                cell_type=cell_type,
                hidden_size=hidden_size,
                encoder_embeddings=self.encoder_embeddings,
                n_layers=n_layers,
                bias=True,
                dropout=dropout,
                return_enc=return_enc,
                lagged_target_positions=self.lagged_target_positions,
            )
        elif encoderdecodertype == "AUTO":
            self.encoderdecoder = AutoencoderType(
                cont_size=len(self._encoder_cont),
                encoder_embeddings=self.encoder_embeddings,
                n_layers=n_layers,
                # dropout=dropout,
                return_enc=return_enc,
                hidden_size=hidden_size,
                sequence_length=steps,
            )
        elif encoderdecodertype == "VAE":
            self.encoderdecoder = VariAutoencoderType(
                cont_size=len(self._encoder_cont),
                encoder_embeddings=self.encoder_embeddings,
                n_layers=n_layers,
                # dropout=dropout,
                return_enc=return_enc,
                hidden_size=hidden_size,
                sequence_length=steps,
            )
        # transform decoder-output

    @cached_property
    def _encoder_input_size(self) -> int:
        """the actual input size/dim of the encoder after the categorical embedding, ignored when not considering cat"""
        return len(self._encoder_cont) + self.encoder_embeddings.total_embedding_size()

    @property
    def reals(self) -> List[str]:
        """lists of reals in the encoder or decoder sequence"""
        return self._encoder_cont

    @property
    def reals_indices(self) -> List[int]:
        """lists of the indices of reals in `x["encoder_cont"]`"""
        return [self._encoder_cont.index(target) for target in self.reals]

    @property
    def categoricals(self) -> List[str]:
        """lists of categoricals in the encoder or decoder sequence"""
        return self._encoder_cat

    @property
    def categoricals_embedding(self) -> MultiEmbedding:
        return self.encoder_embeddings

    @property
    def encoder_positions(self) -> torch.LongTensor:
        """Target positions in the encoder or decoder tensor

        Note that when `time_varying_unknown_reals` is present, `target_positions` gives the indices
        of target columns after dropping `time_varying_unknown_reals` from `x["decoder_cont"]`

        Returns:
            torch.LongTensor: tensor of positions.
        """
        pos = torch.tensor(
            [self._targets.index(name) for name in to_list(self._encoder_cont)],
            dtype=torch.long,
        )
        a = torch.tensor(
            [i for i, p in enumerate(self.reals_indices) if p in pos],
            dtype=torch.long,
        )
        # device=self.device,
        return a

    @property
    def lagged_target_positions(self) -> Dict[int, torch.LongTensor]:
        # todo: expand for categorical targets
        if len(self._targets_lags) == 0:
            pos = {}
        else:
            # extract lags which are the same across all targets
            lags = list(next(iter(self._targets_lags.values())).values())
            lag_names = {l: [] for l in lags}
            for targeti_lags in self._targets_lags.values():
                for name, l in targeti_lags.items():
                    lag_names[l].append(name)

            pos = {
                lag: torch.tensor(
                    [self._encoder_cont.index(name) for name in to_list(names)],
                    dtype=torch.long,
                    device=self._encoder_cont.device,
                )
                for lag, names in lag_names.items()
            }

        return {
            k: (torch.tensor(self.reals_indices, device=self.reals_indices.device) == v).nonzero(
                as_tuple=True
            )[0]
            if len(v) == 1
            else torch.stack(
                [
                    (torch.tensor(self.reals_indices, device=self.reals_indices.device) == item).nonzero(
                        as_tuple=True
                    )[0]
                    for item in v
                ]
            ).nonzero(as_tuple=True)[0]
            for k, v in pos.items()
        }

    def forward(self, x):
        raise NotImplementedError()

    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "BaseModel":
        raise NotImplementedError()

    def calculate_params(self, **kwargs):
        """
        calculate specific post-training parameters in model
        """
        return

    def predict(self, **kwargs):
        """
        calculate specific post-predict parameters in model
        """
        return


class BaseLong(BaseModel):
    def __init__(
        self,
        targets: str,
        output_size: int,
        context_length: int,
        prediction_length: int,
        token_length: int,
        # hpyerparameters
        fcn_size: int = 1024,
        n_heads: int = 0,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
        attn_type: str = "prob",
        # data types
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        decoder_cont: List[str] = [],
        decoder_cat: List[str] = [],
        target_lags: Dict = {},
        freq: str = "h",
    ):
        super().__init__()

        self._targets = targets
        self._output_size = output_size
        self._encoder_cont = encoder_cont
        self._encoder_cat = encoder_cat
        self._decoder_cont = decoder_cont
        self._decoder_cat = decoder_cat
        self._targets_lags = target_lags
        self._prediction_length = prediction_length
        self._token_length = token_length
        self._context_length = context_length
        self._freq = freq
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.fcn_size = fcn_size
        self.attn_type = ProbAttention if attn_type == "prob" else FullAttention
        assert (
            self._targets not in self._decoder_cont
        ), f"Target: {self._targets} should not in decoder_cont, which contains: {self._decoder_cont}"

        # embedding both `cont` and `cat`
        self.enc_embedding = DataEmbedding(
            len(encoder_cont) + len(encoder_cat), hidden_size, self._freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            len(decoder_cont) + len(decoder_cat), hidden_size, self._freq, dropout
        )

        # factor always equal to 3, which uses to determine top-k, activation always equal to gelu
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        self.attn_type(mask_flag=False, attention_dropout=dropout, output_attention=False),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    fcn_size,
                    dropout,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ],
            # last attention dont need cov, so n_cov=n_atten - 1
            conv_layers=[ConvLayer(hidden_size) for _ in range(n_layers - 1)],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )
        self._phase = "encode"
        self.decoder = Decoder(
            attn_layers=[
                DecoderLayer(
                    AttentionLayer(
                        self.attn_type(mask_flag=True, attention_dropout=dropout, output_attention=False),
                        hidden_size,
                        n_heads,
                    ),
                    AttentionLayer(
                        self.attn_type(mask_flag=False, attention_dropout=dropout, output_attention=False),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    fcn_size,
                    dropout,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )

        self.projection = nn.Linear(hidden_size, output_size, bias=True)

    @property
    def cont_size(self):
        return len(self._encoder_cont)

    @property
    def cat_size(self):
        return len(self._encoder_cat)

    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, HIDDENSTATE]:
        """Encode a sequence into hidden state and make backcasting predictions

        Args:
            x (Dict[str, torch.Tensor]): the input dictionary

        Returns:
            Tuple[torch.Tensor, HiddenState]:
                * the prediction on the encoding sequence
                * the last hidden state
        """
        self._phase = "encode"
        input_vector = []
        # NOTE: the real-valued variables always come first in the input vector
        if self.cont_size > 0:
            input_vector.append(x["encoder_cont"].clone())
        if self.cat_size > 0:
            input_vector.append(x["encoder_cat"].clone())

        input_vector = torch.cat(input_vector, dim=-1)
        enc_out = self.enc_embedding(input_vector, x["encoder_time_features"])
        # output_attention = False which is default leading to attns = []
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        return enc_out, attns

    def decode(
        self,
        x: Dict[str, torch.Tensor],
        hidden_state: HIDDENSTATE,
        **_,
    ) -> torch.Tensor:
        """Decode a hidden state and an input sequence into forcasting  predictions.

        Args:
            x (Dict[str, torch.Tensor]): the input dictionary
            hidden_state (HiddenState): the last output from the encoder

        Returns:
            torch.Tensor: the prediction on the decoding sequence
        """
        self._phase = "decode"

        # place token_length encoder_target in the start of decoder serires
        # initialize the rest with zero
        decoder_init = (
            torch.cat(
                [
                    x["encoder_target"][:, self._context_length - self._token_length :, :],
                    torch.zeros_like(x["decoder_target"][:, -self._prediction_length :, :]),
                ],
                dim=1,
            )
            .float()
            .to(x["encoder_target"].device)
        )
        decoder_time_features = torch.cat(
            [
                x["encoder_time_features"][:, self._context_length - self._token_length :, :],
                x["decoder_time_features"],
            ],
            dim=1,
        )

        dec_out = self.dec_embedding(decoder_init, decoder_time_features)
        dec_out = self.decoder(dec_out, hidden_state, x_mask=None, cross_mask=None)

        return dec_out[:, -self._prediction_length :, :]
