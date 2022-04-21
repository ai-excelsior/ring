from lib2to3.pytree import Base
from tkinter import HIDDEN
from torch import nn
import torch
from .dataset import TimeSeriesDataset
from functools import cached_property
from typing import Any, Callable, Dict, List, Tuple, Union
from ring.common.ml.embeddings import MultiEmbedding
from ring.common.ml.rnn import get_rnn
from ring.common.ml.utils import to_list
import numpy as np
import random
from torch.autograd import Variable

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
            if not self._decoder_cat == self._encoder_cat
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
            #  device=self.device,
            dtype=torch.long,
        )
        a = torch.tensor(
            [i for i, p in enumerate(self.reals_indices) if p in pos],
            # device=self.device,
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
            lag_names = {l: [] for l in lags}
            for targeti_lags in self._targets_lags.values():
                for name, l in targeti_lags.items():
                    lag_names[l].append(name)

            pos = {
                lag: torch.tensor(
                    [self._encoder_cont.index(name) for name in to_list(names)],
                    dtype=torch.long,
                )
                for lag, names in lag_names.items()
            }

        return {
            k: (torch.tensor(self.reals_indices) == v).nonzero(as_tuple=True)[0]
            if len(v) == 1
            else torch.stack(
                [(torch.tensor(self.reals_indices) == item).nonzero(as_tuple=True)[0] for item in v]
            ).nonzero(as_tuple=True)[0]
            for k, v in pos.items()
        }

    @cached_property
    def has_time_varying_unknown_cont(self) -> bool:
        return set(self._encoder_cont) != set(self._decoder_cont)

    @cached_property
    def has_time_varying_unknown_cat(self) -> bool:
        return set(self._encoder_cat) != set(self._decoder_cat)

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
        normalized_target = [input_vector[:, 0, self.target_positions]]
        # the autoregression loop
        for idx in range(n_decoder_steps):
            _input_vector = input_vector[:, [idx]]
            # take the last predicted target values as the input for the current prediction step
            _input_vector[:, 0, self.target_positions] = normalized_target[-1]
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
        input_vector[..., self.target_positions].roll(shifts=1, dims=1)

        if first_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions] = first_target
        else:  # or drop the first time step
            input_vector = input_vector[:, 1:]
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
            output,_ = self._decode(
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


##########################
class BaseAnormal(BaseModel):
    def __init__(
        self,
        name: str = None,
        # hpyerparameters
        # data types
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        x_categoricals: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        target_lags: Dict = {},
    ):
        super().__init__()

        self._encoder_cont = encoder_cont
        self._encoder_cat = encoder_cat
        self._x_categoricals = x_categoricals
        self._targets_lags = target_lags

        self.encoder_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=[],
            categorical_groups={},
            x_categoricals=x_categoricals,
        )

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
                )
                for lag, names in lag_names.items()
            }

        return {
            k: (torch.tensor(self.reals_indices) == v).nonzero(as_tuple=True)[0]
            if len(v) == 1
            else torch.stack(
                [(torch.tensor(self.reals_indices) == item).nonzero(as_tuple=True)[0] for item in v]
            ).nonzero(as_tuple=True)[0]
            for k, v in pos.items()
        }

    def forward(self, X):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "BaseModel":
        raise NotImplementedError()

    def construct_input_vector(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        reverse: torch.Tensor = False,
    ) -> torch.Tensor:
        # create input vector
        input_vector = []
        # NOTE: the real-valued variables always come first in the input vector
        if len(self.reals) > 0:
            input_vector.append(x_cont.clone())

        if len(self.categoricals) > 0:
            embeddings = self.categoricals_embedding(x_cat, flat=True)
            input_vector.append(embeddings)

        input_vector = torch.cat(input_vector, dim=-1)
        if reverse:  # predict from backward
            input_vector = input_vector.flip(1)  # reverse the input_vector by row

        return input_vector

    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, HIDDENSTATE]:
        encoder_cat, encoder_cont, lengths = x["encoder_cat"], x["encoder_cont"], x["encoder_length"] - 1
        assert lengths.min() > 0
        input_vector = self.construct_input_vector(encoder_cat, encoder_cont)  # concat cat and cont
        _, hidden_state = self.encoder(input_vector, lengths=lengths, enforce_sorted=False)
        return hidden_state

    def decode(
        self,
        x: Dict[str, torch.Tensor] = {},
        hidden_state: HIDDENSTATE = None,
        **_,
    ) -> torch.Tensor:
        encoder_cat, encoder_cont = x["encoder_cat"], x["encoder_cont"]
        input_vector = self.construct_input_vector(encoder_cat, encoder_cont, reverse=True)
        if self.training:  # the training mode where the target values are actually known
            output, _ = self._decode(
                input_vector=input_vector, hidden_state=hidden_state, lengths=x["encoder_length"]
            )
        else:  # the prediction mode, in which the target values are unknown
            output = self.decode_autoregressive(
                self._decode,
                input_vector=input_vector,
                hidden_state=hidden_state,
                n_decoder_steps=x["encoder_length"][0],
            )
        return output.flip(1)  # reverse order

    def _decode(
        self,
        input_vector: torch.Tensor,
        hidden_state: HIDDENSTATE,
        lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, HIDDENSTATE]:

        if self.cell_type == "LSTM":
            last_value = hidden_state[0][-1, ...]
        else:
            last_value = hidden_state[-1, ...]
        # {p0:h1, p1:h2, p2:h3 ... pt:ht+1}
        output, hidden_state_ = self.decoder(
            input_vector, hidden_state, lengths=lengths, enforce_sorted=False
        )
        if self.training:
            # roll `ht+1` to p0 and adjust rest positions of `h`
            output.roll(dims=1, shifts=1)
            # replace p0:ht+1 with p0:h0
            output[:, 0] = last_value

        return self.output_projector_decoder(output), hidden_state_

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
        # take the first predicted target from the projection of last layer of `hidden_state`
        if self.cell_type == "LSTM":
            normalized_target = [self.output_projector_decoder(hidden_state[0][-1, ...])]
        else:
            normalized_target = [self.output_projector_decoder(hidden_state[-1, ...])]
        # the autoregression loop
        for idx in range(n_decoder_steps):
            # batch_size * 1 * n_features
            _input_vector = input_vector[:, [idx]]
            # take the last predicted target values as the input for the current prediction step
            _input_vector[:, 0, :] = normalized_target[-1]
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
            normalized_target.append(output)
            predictions.append(output)

        return torch.stack(predictions, dim=1)