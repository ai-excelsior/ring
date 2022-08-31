"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""
from copy import copy
from typing import Dict, List, Tuple, Union
from tomlkit import TOMLDocument
from collections import namedtuple

import torch
from torch import nn
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from pytorch_forecasting.utils import create_mask, OutputMixIn

from ring.common.base_model import AutoRegressiveBaseModelWithCovariates
from ring.common.dataset import TimeSeriesDataset


class TemporalFusionTransformer(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        targets: str,
        # hyper parameters
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 7,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        # data parameters
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        encoder_cont: List[str] = [],
        encoder_cat: List[str] = [],
        decoder_cont: List[str] = [],
        decoder_cat: List[str] = [],
        x_categoricals: List[str] = [],
        # network parameters
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        share_single_variable_networks: bool = False,
        **kwargs,
    ):
        """
        Temporal Fusion Transformer for forecasting timeseries - use its :py:meth:`~from_dataset` method if possible.

        Implementation of the article
        `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
        Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_. The network outperforms DeepAR by Amazon by 36-69%
        in benchmarks.

        Enhancements compared to the original implementation (apart from capabilities added through base model
        such as monotone constraints):

        * static variables can be continuous
        * multiple categorical variables can be summarized with an EmbeddingBag
        * variable encoder and decoder length by sample
        * categorical embeddings are not transformed by variable selection network (because it is a redundant operation)
        * variable dimension in variable selection network are scaled up via linear interpolation to reduce
          number of parameters
        * non-linear variable processing in variable selection network can be shared among decoder and encoder
          (not shared by default)

        Tune its hyperparameters with
        :py:func:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`.

        Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes).
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            share_single_variable_networks (bool): if to share the single variable networks between the encoder and
                decoder. Defaults to False.
            logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """

        super().__init__(
            targets=targets,
            cell_type="LSTM",
            hidden_size=hidden_size,
            n_layers=lstm_layers,
            dropout=dropout,
            encoder_cont=encoder_cont,
            encoder_cat=encoder_cat,
            decoder_cont=decoder_cont,
            decoder_cat=decoder_cat,
            embedding_sizes=embedding_sizes,
            x_categoricals=x_categoricals,
            # target_lags=target_lags,
            output_size=output_size,
        )

        self.lstm_layers = (lstm_layers,)
        self.output_size = (output_size,)

        # data parameters
        self.static_categoricals = (static_categoricals,)
        self.static_reals = (static_reals,)
        self.time_varying_categoricals_encoder = (time_varying_categoricals_encoder,)
        self.time_varying_categoricals_decoder = (time_varying_categoricals_encoder,)
        self.time_varying_reals_encoder = (time_varying_reals_encoder,)
        self.time_varying_reals_decoder = (time_varying_reals_decoder,)

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            categorical_groups={},
            embedding_paddings=[],
            x_categoricals=x_categoricals,
            max_embedding_size=hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(
                    1,
                    hidden_continuous_sizes.get(name, hidden_continuous_size),
                )
                for name in self.reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {name: embedding_sizes[name][1] for name in static_categoricals}
        static_input_sizes.update(
            {name: hidden_continuous_sizes.get(name, hidden_continuous_size) for name in static_reals}
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=hidden_size,
            input_embedding_flags={name: True for name in static_categoricals},
            dropout=dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {name: embedding_sizes[name][1] for name in time_varying_categoricals_encoder}
        encoder_input_sizes.update(
            {
                name: hidden_continuous_sizes.get(name, hidden_continuous_size)
                for name in time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {name: embedding_sizes[name][1] for name in time_varying_categoricals_decoder}
        decoder_input_sizes.update(
            {
                name: hidden_continuous_sizes.get(name, hidden_continuous_size)
                for name in time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, hidden_size),
                    hidden_size,
                    dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, hidden_size),
                        hidden_size,
                        dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=hidden_size,
            input_embedding_flags={name: True for name in time_varying_categoricals_encoder},
            dropout=dropout,
            context_size=hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=hidden_size,
            input_embedding_flags={name: True for name in time_varying_categoricals_decoder},
            dropout=dropout,
            context_size=hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            hidden_size,
            hidden_size,
            hidden_size,
            dropout,
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_lstm_add_norm_encoder = AddNorm(hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            n_head=attention_head_size,
            dropout=dropout,
        )
        self.post_attn_gate_norm = GateAddNorm(hidden_size, dropout=dropout, trainable_add=False)
        self.pos_wise_ff = GatedResidualNetwork(
            hidden_size,
            hidden_size,
            hidden_size,
            dropout=dropout,
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(hidden_size, dropout=None, trainable_add=False)

        if output_size > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [nn.Linear(hidden_size, output_size) for output_size in output_size]
            )
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataset,
        # allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        # add maximum encoder length
        # update defaults
        # new_kwargs = copy(kwargs)
        # new_kwargs["max_encoder_length"] = kwargs["max_encoder_length"]
        # new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # create class and return

        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = {}
        for k, v in dataset.embedding_sizes.items():
            if k in dataset.encoder_cat:
                embedding_sizes[k] = v
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)
        kwargs.setdefault(
            "target_lags",
            {lag.feature_name: lag._state for lag in dataset.lags},
        )
        return cls(
            targets=dataset.targets,
            encoder_cat=dataset.encoder_cat,
            encoder_cont=dataset.encoder_cont + dataset.time_features + dataset.encoder_lag_features,
            decoder_cat=dataset.decoder_cat,
            decoder_cont=dataset.decoder_cont + dataset.time_features + dataset.decoder_lag_features,
            embedding_sizes=embedding_sizes,
            x_categoricals=dataset.categoricals,
            static_categoricals=dataset._static_categoricals,
            static_reals=dataset._static_reals,
            time_varying_categoricals_encoder=dataset._time_varying_known_categoricals,
            time_varying_categoricals_decoder=dataset._time_varying_unknown_categoricals,
            time_varying_reals_encoder=dataset._time_varying_unknown_reals,
            time_varying_reals_decoder=dataset._time_varying_unknown_reals,  # TODO confirm
            x_reals=dataset.reals,
            **kwargs,
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]

        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask

    def static_variables(self) -> List[str]:
        """List of all static variables in model"""
        return self.static_categoricals + self.static_reals

    def encoder_variables(self) -> List[str]:
        """List of all encoder variables in model (excluding static variables)"""
        return self.time_varying_categoricals_encoder + self.time_varying_reals_encoder

    @property
    def decoder_variables(self) -> List[str]:
        """List of all decoder variables in model (excluding static variables)"""
        return self.time_varying_categoricals_decoder + self.time_varying_reals_decoder

    def to_network_output(self, **results):
        """
        Convert output into a named (and immuatable) tuple.

        This allows tracing the modules as graphs and prevents modifying the output.

        Returns:
            named tuple
        """
        if hasattr(self, "_output_class"):
            Output = self._output_class
        else:
            OutputTuple = namedtuple("output", results)

            class Output(OutputMixIn, OutputTuple):
                pass

            self._output_class = Output

        return self._output_class(**results)

    # def transform_output(
    #     self,
    #     prediction: Union[torch.Tensor, List[torch.Tensor]],
    #     target_scale: Union[torch.Tensor, List[torch.Tensor]],
    # ) -> torch.Tensor:
    #     """
    #     Extract prediction from network output and rescale it to real space / de-normalize it.

    #     Args:
    #         prediction (Union[torch.Tensor, List[torch.Tensor]]): normalized prediction
    #         target_scale (Union[torch.Tensor, List[torch.Tensor]]): scale to rescale prediction

    #     Returns:
    #         torch.Tensor: rescaled prediction
    #     """
    #     if isinstance(self.loss, MultiLoss):
    #         out = self.loss.rescale_parameters(
    #             prediction,
    #             target_scale=target_scale,
    #             encoder=self.output_transformer.normalizers,  # need to use normalizer per encoder
    #         )
    #     else:
    #         out = self.loss.rescale_parameters(
    #             prediction, target_scale=target_scale, encoder=self.output_transformer
    #         )
    #     return out

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_length"]
        decoder_lengths = x["decoder_length"]

        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        # x_cont = torch.cat(
        #     [x["encoder_cont"] - self._targets, x["decoder_cont"]], dim=1
        # )  # concatenate in time dimension
        x_cont = x["encoder_cont"]
        timesteps = x_cont.shape[1]  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.reals)
                if name in self.reals
            }
        )
        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            (
                static_embedding,
                static_variable_selection,
            ) = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding),
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        (embeddings_varying_encoder, encoder_sparse_weights,) = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]
            for name in self.decoder_variables  # select decoder
        }
        (embeddings_varying_decoder, decoder_sparse_weights,) = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths,
                decoder_length=timesteps - max_encoder_length,
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.output_size > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)

        return self.to_network_output(
            # prediction=self.transform_output(output, target_scale=x["target_scale"]),
            prediction=output,
            attention=attn_output_weights,
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )