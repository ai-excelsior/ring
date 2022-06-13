from copy import deepcopy
from torch import nn
import torch.nn.functional as F
import torch
from typing import Callable, Dict, List, Tuple, Union
from ring.common.ml.embeddings import MultiEmbedding
from ring.common.ml.rnn import get_rnn
from ring.common.mask import ProbMask, TriangularCausalMask
import numpy as np


HIDDENSTATE = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseType(nn.Module):
    def __init__(
        self,
        name: str = "Base",
        cont_size: int = 1,
        encoder_embeddings: MultiEmbedding = None,
    ):
        super().__init__()
        self.cont_size = cont_size
        self.encoder_embeddings = encoder_embeddings

    def construct_input_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, reverse: bool = False
    ) -> torch.Tensor:
        # create input vector
        input_vector = []
        # NOTE: the real-valued variables always come first in the input vector
        if self.cont_size > 0:
            input_vector.append(x_cont.clone())

        if self.encoder_embeddings.total_embedding_size() > 0:
            # embeddings = self.encoder_embeddings(x_cat.clone(), flat=True)
            embeddings = x_cat.clone()
            input_vector.append(embeddings)

        input_vector = torch.cat(input_vector, dim=-1)
        if reverse:  # predict from backward
            input_vector = input_vector.flip(1)  # reverse the input_vector by row

        return input_vector


class RnnType(BaseType):
    def __init__(
        self,
        name: str = "RNN_type",
        cont_size: int = 1,
        # hpyerparameters
        # data types
        cell_type: str = "LSTM",
        hidden_size: int = 8,
        n_layers: int = 1,
        dropout: float = 0,
        encoder_embeddings: MultiEmbedding = None,
        bias: bool = True,
        return_enc: bool = False,
        lagged_target_positions: Dict = {},
    ):

        super().__init__(name=name, cont_size=cont_size, encoder_embeddings=encoder_embeddings)
        self.lagged_target_positions = lagged_target_positions
        self.return_enc = return_enc
        self.cell_type = cell_type

        self.encoder = get_rnn(cell_type)(
            input_size=cont_size + encoder_embeddings.total_embedding_size(),
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layers,
            bias=bias,
            dropout=dropout,
        )

        self.decoder = get_rnn(cell_type)(
            input_size=cont_size + encoder_embeddings.total_embedding_size(),
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layers,
            bias=bias,
            dropout=dropout,
        )
        self.output_projector_decoder = nn.Linear(
            hidden_size, cont_size + encoder_embeddings.total_embedding_size()
        )

    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, HIDDENSTATE]:
        encoder_cat, encoder_cont = x["encoder_cat"], x["encoder_cont"]
        lengths = x["encoder_length"] - 1  # skip last timestamp
        assert lengths.min() > 0
        input_vector = self.construct_input_vector(encoder_cat, encoder_cont)  # concat cat and cont
        output, hidden_state = self.encoder(input_vector, lengths=lengths, enforce_sorted=False)
        return output, hidden_state

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
            output = output.roll(dims=1, shifts=1)
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
        # take the first predicted target from the projection of last layer of `hidden_state`
        if self.cell_type == "LSTM":
            predictions = [self.output_projector_decoder(hidden_state[0][-1, ...])]
        else:
            predictions = [self.output_projector_decoder(hidden_state[-1, ...])]
        # the autoregression loop
        for idx in range(n_decoder_steps - 1):
            # batch_size * 1 * n_features
            _input_vector = input_vector[:, [idx]]
            # take the last predicted target values as the input for the current prediction step
            _input_vector[:, 0, :] = predictions[-1]
            for lag, lag_positions in self.lagged_target_positions.items():
                # lagged values are depleted: if the current prediction step is beyond the lag
                if idx > lag and len(lag_positions) > 0:
                    _input_vector[:, 0, lag_positions] = predictions[-lag]

            output_, hidden_state = decode_one_step(
                input_vector=_input_vector,
                hidden_state=hidden_state,
                **kwargs,
            )

            output = [o.squeeze(1) for o in output_] if isinstance(output_, list) else output_.squeeze(1)

            predictions.append(output)
        pre = torch.stack(predictions, dim=1)  # row

        return pre

    def forward(self, x):
        enc_output, enc_hidden = self.encode(x)
        simulation = self.decode(x, hidden_state=enc_hidden)
        if self.return_enc:
            return enc_output, simulation
        else:
            return simulation


class AutoencoderType(BaseType):
    def __init__(
        self,
        name: str = "RNN_type",
        cont_size: int = 1,
        # hpyerparameters
        n_layers: int = None,
        encoder_embeddings: MultiEmbedding = None,
        return_enc: bool = False,
        hidden_size: int = 2,
        sequence_length: int = 1,
    ):
        super().__init__(name=name, cont_size=cont_size, encoder_embeddings=encoder_embeddings)
        self.return_enc = return_enc
        self.cont_size = cont_size
        self.n_layers = n_layers
        self.encoder_embeddings = encoder_embeddings

        # flatten
        initial_size = sequence_length * (cont_size + encoder_embeddings.total_embedding_size())
        # encoder_layer += [nn.Tanh()]
        encoder_layer = []
        encoder_layer += [nn.Linear(118, 60)]
        encoder_layer += [nn.Tanh()]
        encoder_layer += [nn.Linear(60, 30)]
        encoder_layer += [nn.Tanh()]
        encoder_layer += [nn.Linear(30, 10)]
        encoder_layer += [nn.Tanh()]
        encoder_layer += [nn.Linear(10, 1)]
        # if not self.n_layers:
        #     hidden_list = (
        #         2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(initial_size))[1:][::-1]
        #     )
        # else:
        #     hidden_list = [int(initial_size // (2 ** i)) for i in range(n_layers)][1:]
        # # encoder part
        # encoder_layer = [nn.Linear(initial_size, hidden_list[0]), nn.Tanh()]
        # for i in range(len(hidden_list) - 1):
        #     encoder_layer.extend([nn.Linear(hidden_list[i], hidden_list[i + 1]), nn.Tanh()])
        # encoder_layer.extend([nn.Linear(hidden_list[i + 1], hidden_size)])
        self.encoder = nn.Sequential(*encoder_layer)
        # # decoder part
        # decoder_layer = [nn.Linear(hidden_size, hidden_list[-1]), nn.Tanh()]
        # for i in range(len(hidden_list) - 1, 0, -1):
        #     decoder_layer.extend([nn.Linear(hidden_list[i], hidden_list[i - 1]), nn.Tanh()])
        # decoder_layer.extend([nn.Linear(hidden_list[i - 1], initial_size)])

        decoder_layer = []
        decoder_layer += [nn.Linear(1, 10)]
        decoder_layer += [nn.Tanh()]
        decoder_layer += [nn.Linear(10, 30)]
        decoder_layer += [nn.Tanh()]
        decoder_layer += [nn.Linear(30, 60)]
        decoder_layer += [nn.Tanh()]
        decoder_layer += [nn.Linear(60, 118)]
        self.decoder = nn.Sequential(*decoder_layer)

    def forward(self, x):
        encoder_cat, encoder_cont = x["encoder_cat"], x["encoder_cont"]
        input_vector = self.construct_input_vector(encoder_cat, encoder_cont)  # concat cat and cont

        flattened_input = input_vector.view(input_vector.size(0), -1)
        enc = self.encoder(flattened_input)
        dec = self.decoder(enc)

        # `AutoencoderType` dont have `layers`
        # so `enc`::batch_size * hidden_size should be unsqueezed to match the output format of `RnnType`::batch_size * layers * hidden_size
        # `input_vector` have been flattened to batch_size * (steps * features)
        # so `dec`::batch_size * (steps * features) should be reshape to match the reconstruction format of `RnnType`::batch_size * steps * features
        if self.return_enc:
            return enc.unsqueeze(1), dec.view(input_vector.size())
        else:
            return dec


class VariAutoencoderType(AutoencoderType):
    def __init__(
        self,
        name: str = "VAE",
        cont_size: int = 1,
        # hpyerparameters
        n_layers: int = None,
        encoder_embeddings: MultiEmbedding = None,
        return_enc: bool = False,
        hidden_size: int = 2,
        sequence_length: int = 1,
        # dropout: float = 0,
    ):
        super().__init__(
            name=name,
            cont_size=cont_size,
            encoder_embeddings=encoder_embeddings,
            return_enc=return_enc,
            n_layers=n_layers,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
        )
        self.return_enc = return_enc
        self.cont_size = cont_size
        self.n_layers = n_layers
        self.encoder_embeddings = encoder_embeddings
        # sampling net

        # flatten
        initial_size = sequence_length * (cont_size + encoder_embeddings.total_embedding_size())

        if not self.n_layers:
            hidden_list = (
                2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(initial_size))[1:][::-1]
            )
        else:
            hidden_list = [int(initial_size // (2 ** i)) for i in range(n_layers)][1:]

        self.mean_layer = nn.Linear(hidden_list[-1], hidden_size)
        self.var_layer = nn.Linear(hidden_list[-1], hidden_size)
        # encoder part
        encoder_layer = [nn.Linear(initial_size, hidden_list[0]), nn.Tanh()]
        for i in range(len(hidden_list) - 1):
            encoder_layer.extend([nn.Linear(hidden_list[i], hidden_list[i + 1]), nn.Tanh()])
        self.encoder = nn.Sequential(*encoder_layer[:-1])  # remove last tanh
        # decoder part
        decoder_layer = [nn.Linear(hidden_size, hidden_list[-1]), nn.Tanh()]
        for i in range(len(hidden_list) - 1, 0, -1):
            decoder_layer.extend([nn.Linear(hidden_list[i], hidden_list[i - 1]), nn.Tanh()])
        decoder_layer.extend([nn.Linear(hidden_list[i - 1], initial_size)])
        self.decoder = nn.Sequential(*decoder_layer)

    def sampling(self, latent):
        z_mean = self.mean_layer(latent)
        z_var = self.var_layer(latent)
        eps = torch.randn_like(z_mean).to(latent.device)
        kl_loss = -0.5 * torch.sum(1 + z_var - torch.square(z_mean) - torch.exp(z_var), axis=1)
        return z_mean + torch.exp(0.5 * z_var) * eps, kl_loss

    def forward(self, x):
        encoder_cat, encoder_cont = x["encoder_cat"], x["encoder_cont"]
        input_vector = self.construct_input_vector(encoder_cat, encoder_cont)  # concat cat and cont

        flattened_input = input_vector.view(input_vector.size(0), -1)
        # batch_size * hidden_list[-1]
        enc = self.encoder(flattened_input)
        # batch_size * hidden_size
        z, kl_loss = self.sampling(enc)

        dec = self.decoder(z)
        # `AutoencoderType` dont have `layers`
        # so `enc`::batch_size * hidden_size should be unsqueezed to match the output format of `RnnType`::batch_size * layers * hidden_size
        # `input_vector` have been flattened to batch_size * (steps * features)
        # so `dec`::batch_size * (steps * features) should be reshape to match the reconstruction format of `RnnType`::batch_size * steps * features
        if self.return_enc:
            return z.unsqueeze(1), (dec.view(input_vector.size()), kl_loss)
        else:
            return (dec, kl_loss)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, hidden_size, fcn_size=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        fcn_size = fcn_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=fcn_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=fcn_size, out_channels=hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:  # n_layers >1
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                # go through a attention layer
                x, attn = attn_layer(x, attn_mask=attn_mask)
                # go through a cov layer
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders):
        super(EncoderStack, self).__init__()
        # decrease 1 conv layer each stack
        # for i in range(1, len(encoders)):
        #     encoders[i] = deepcopy(encoders[i])
        #     encoders[i].conv_layers = encoders[i].conv_layers[:-i]
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, attn_mask=None):
        # x [batch_size, look_back, n_target]
        x_stack = []
        attns = []
        # n_stacks
        for i in range(len(self.encoders)):
            # half the input for each stack
            inp_len = x.shape[1] // (2 ** i)
            x_s, attn = self.encoders[i](x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)

        # concat all-stack output
        x_stack = torch.cat(x_stack, -2)
        return x_stack, attns


class DecoderLayer(nn.Module):
    def __init__(
        self, self_attention, cross_attention, hidden_size, fcn_size=None, dropout=0.1, activation="relu"
    ):
        super(DecoderLayer, self).__init__()
        fcn_size = fcn_size or 4 * hidden_size
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=fcn_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=fcn_size, out_channels=hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.attn_layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
