import torch
from torch import nn
from typing import List, Dict, Tuple, Union
from copy import deepcopy, copy

from ring.common.base_model import BaseModel
from ring.common.ml.rnn import get_rnn
from ring.common.ml.embeddings import MultiEmbedding
from ring.common.dataset import TimeSeriesDataset
from ring.common.normalizers import AbstractNormalizer
from ring.ts.nbeats.submodules import NBEATSBlock, NBEATSGenericBlock, NBEATSSeasonalBlock, NBEATSTrendBlock


class NbeatsNetwork(BaseModel):
    def __init__(
        self,
        targets: str,
        model_type: str = "G",  # 'I'
        num_stack: int = 1,
        num_block: int = 3,
        width: int = 8,  # [2**8]
        expansion_coe: int = 5,  # [3,7]
        num_block_layer: int = 4,
        prediction_length: int = 0,
        context_length: int = 0,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0,
        target_number: int = 1,
        covariate_number: int = 0,
        encoder_cont: List[str] = [],
        decoder_cont: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        x_categoricals: List[str] = [],
    ):

        super().__init__()
        self._targets = targets
        self._encoder_cont = encoder_cont
        self._decoder_cont = decoder_cont
        self.dropout = dropout
        self.backcast_loss_ratio = backcast_loss_ratio
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_number = target_number
        self.covariate_number = covariate_number

        self.encoder_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=[],
            categorical_groups={},
            x_categoricals=x_categoricals,
        )

        if model_type == "I":
            width = [2 ** width, 2 ** (width + 2)]
            self.stack_types = ["trend", "seasonality"] * num_stack
            self.expansion_coefficient_lengths = [item for i in range(num_stack) for item in [3, 7]]
            self.num_blocks = [num_block for i in range(2 * num_stack)]
            self.num_block_layers = [num_block_layer for i in range(2 * num_stack)]
            self.widths = [item for i in range(num_stack) for item in width]
        elif model_type == "G":
            self.stack_types = ["generic"] * num_stack
            self.expansion_coefficient_lengths = [2 ** expansion_coe for i in range(num_stack)]
            self.num_blocks = [num_block for i in range(num_stack)]
            self.num_block_layers = [num_block_layer for i in range(num_stack)]
            self.widths = [2 ** width for i in range(num_stack)]
        #
        # setup stacks
        self.net_blocks = nn.ModuleList()

        for stack_id, stack_type in enumerate(self.stack_types):
            for _ in range(self.num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = NBEATSGenericBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                        tar_num=self.target_number,
                        cov_num=self.covariate_number + self.encoder_embeddings.total_embedding_size(),
                        tar_pos=self.target_positions,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlock(
                        units=self.widths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        min_period=self.expansion_coefficient_lengths[stack_id],
                        dropout=self.dropout,
                        tar_num=self.target_number,
                        cov_num=self.covariate_number + self.encoder_embeddings.total_embedding_size(),
                        tar_pos=self.target_positions,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                        tar_num=self.target_number,
                        cov_num=self.covariate_number + self.encoder_embeddings.total_embedding_size(),
                        tar_pos=self.target_positions,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)

    @property
    def target_positions(self):
        return [self._encoder_cont.index(tar) for tar in self._targets]

    def forward(self, x: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Pass forward of network.

        Args:
            x (Dict[str, torch.Tensor]): input from dataloader generated from
                :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Returns:
            Dict[str, torch.Tensor]: output of model
        """
        # batch_size * look_back * features
        encoder_cont = x["encoder_cont"]
        # `target` can only be continuous, so position inside `encoder_cat` is irrelevant
        encoder_cat = (
            torch.cat([v for _, v in self.encoder_embeddings(x["encoder_cat"]).items()], dim=-1)
            if self.encoder_embeddings.total_embedding_size() != 0
            else torch.zeros(
                encoder_cont.size(0),
                self.context_length,
                self.encoder_embeddings.total_embedding_size(),
            )
        )
        # self.hparams.prediction_length=gap+real_predict
        timesteps = self.context_length + self.prediction_length
        # encoder_cont.size(2) + self.encoder_embeddings.total_embedding_size(),
        generic_forecast = [
            torch.zeros(
                (encoder_cont.size(0), timesteps, len(self.target_positions)),
                dtype=torch.float32,
                device=encoder_cont.device,
            )
        ]

        trend_forecast = [
            torch.zeros(
                (encoder_cont.size(0), timesteps, len(self.target_positions)),
                dtype=torch.float32,
                device=encoder_cont.device,
            )
        ]
        seasonal_forecast = [
            torch.zeros(
                (encoder_cont.size(0), timesteps, len(self.target_positions)),
                dtype=torch.float32,
                device=encoder_cont.device,
            )
        ]

        forecast = torch.zeros(
            (encoder_cont.size(0), self.prediction_length, len(self.target_positions)),
            dtype=torch.float32,
            device=encoder_cont.device,
        )

        # make sure `encoder_cont` is followed by `encoder_cat`

        backcast = torch.cat([encoder_cont, encoder_cat], dim=-1)

        for i, block in enumerate(self.net_blocks):
            # evaluate block
            backcast_block, forecast_block = block(backcast)
            # add for interpretation
            full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
            if isinstance(block, NBEATSTrendBlock):
                trend_forecast.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)
            # update backcast and forecast
            backcast = backcast.clone()
            backcast[..., self.target_positions] = backcast[..., self.target_positions] - backcast_block
            # do not use backcast -= backcast_block as this signifies an inline operation
            forecast = forecast + forecast_block

        # `encoder_cat` always at the end of sequence, so it will not affect `self.target_positions`
        # backcast, forecast is of batch_size * context_length/prediction_length * tar_num
        return {
            "prediction": forecast,
            "backcast": (
                encoder_cont[..., self.target_positions] - backcast,
                self.backcast_loss_ratio,
            ),
        }

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "NbeatsNetwork":
        """
        Convenience function to create network from :py:class`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to ``__init__`` method.

        Returns:
            NBeats
        """
        # assert dataset.max_encoder_length%dataset.max_prediction_length==0 and dataset.max_encoder_length<=10*dataset.max_prediction_length,"look back length should be 1-10 times of prediction length"
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = deepcopy(dataset.embedding_sizes)
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)

        return cls(
            dataset.targets,
            encoder_cont=dataset.encoder_cont,
            decoder_cont=dataset.decoder_cont,
            embedding_sizes=embedding_sizes,
            target_number=dataset.n_targets,
            # only for cont, cat will be added in __init__
            covariate_number=len(dataset.encoder_cont) - dataset.n_targets,
            x_categoricals=dataset.categoricals,
            context_length=dataset.get_parameters().get("indexer").get("params").get("look_back"),
            prediction_length=dataset.get_parameters().get("indexer").get("params").get("look_forward"),
        )
