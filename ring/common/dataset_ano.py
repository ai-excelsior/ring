from audioop import minmax
import pandas as pd
import numpy as np
import torch
import inspect
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from ring.common.time_features import time_feature
from ring.common.data_config import DataConfig
from .indexer import BaseIndexer, create_indexer_from_cfg, serialize_indexer, deserialize_indexer
from .normalizers import (
    GroupMinMaxNormalizer,
    MinMaxNormalizer,
    serialize_normalizer,
    deserialize_normalizer,
)
from .encoder import LabelEncoder, PlainEncoder, deserialize_encoder, serialize_encoder
from torch.utils.data.sampler import RandomSampler
from .utils import get_default_embedding_size, add_time_idx
from .encoder import create_encoder_from_cfg

STATIC_UNKNOWN_REAL_NAME = "_ZERO_"
TIME_IDX = "_time_idx_"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        time: str,
        freq: str,
        indexer: BaseIndexer,
        group_ids: List[str] = [],
        # 支持的协变量们
        cat_feature: List[str] = [],
        cont_feature: List[str] = [],
        embedding_sizes: Dict[str, Tuple[str, str]] = None,
        # normalizers
        categorical_encoders: List[PlainEncoder] = [],
        cont_scalars: List[MinMaxNormalizer] = [],
        time_features: List = [],
        # toggles
        last_only=False,  # using last_only mode to index
        start_index=None,  # specify start point to detect
        # seasonality
        lags: Dict = {},
    ) -> None:
        self._time = time
        self._freq = freq
        self._group_ids = group_ids
        self._indexer = indexer
        self._embedding_sizes = embedding_sizes or {}
        self._categorical_encoders = categorical_encoders
        self._cont_scalars = cont_scalars
        self._lags = lags
        self._static_categoricals = []
        self._cat_feature = cat_feature
        self._cont_feature = cont_feature

        # `time_features` will not be added in `decoder_cont` or `encoder_cont`
        # if need, it will be processed in model
        if time_features and not sum([item not in data.columns for item in time_features]):
            self._time_features = time_features
        else:
            time_feature_data = time_feature(pd.to_datetime(data[self._time].values), freq=self._freq)
            self._time_features = list(time_feature_data.columns)
            data = pd.concat([data, time_feature_data], axis=1)

        # add `groupd_ids` to _static_categoricals
        for group_id in group_ids:
            self._static_categoricals.append(group_id)

        # categorical encoder
        if len(self._categorical_encoders) == 0:
            for cat in self.categoricals:  # `group_id` already in
                self._categorical_encoders.append(LabelEncoder(feature_name=cat))

        # cont scalar
        if len(self._cont_scalars) == 0:
            for cont in self.encoder_cont:
                if len(self._group_ids) > 0:
                    self._cont_scalars.append(
                        GroupMinMaxNormalizer(group_ids=self._group_ids, feature_name=cont, center=False)
                    )
                else:
                    self._cont_scalars.append(MinMaxNormalizer(feature_name=cont, center=False))

        assert all(
            group_id in data.columns for group_id in group_ids
        ), f"Please make sure all {group_ids} is in the data"

        # create `embedding_size` for those unassigned
        for categorical_name in self.categoricals:
            if categorical_name not in self._embedding_sizes:
                cat_size = len(data[categorical_name].unique())
                self._embedding_sizes[categorical_name] = (cat_size + 1, get_default_embedding_size(cat_size))

        # initialize indexer
        data = add_time_idx(data, time_column_name=time, freq=freq)
        data.sort_values([*self._group_ids, TIME_IDX], inplace=True)
        data.reset_index(drop=True, inplace=True)
        self._indexer.index(data, last_only, start_index)

        # convert `categoricals`
        for i, cat in enumerate(self.categoricals):
            encoder = self._categorical_encoders[i]
            if not encoder.fitted:
                data[cat] = encoder.fit_transform(data[cat].astype(str))

            else:
                data[cat] = encoder.transform(data[cat].astype(str), self.embedding_sizes)

        # convert `cont`
        for i, cont in enumerate(self.encoder_cont):
            scalar = self._cont_scalars[i]
            if not scalar.fitted:
                data[cont] = scalar.fit_transform(data[cont], data)
            else:
                data[cont] = scalar.transform(data[cont], data)

        self._data = data

    @property
    def targets(self):
        return self._cont_feature + self._cat_feature  # make sure `cat` comes after `cont`

    @property
    def n_targets(self):
        return len(self._cont_feature + self._cat_feature)  # make sure `cat` comes after `cont`

    @property
    def embedding_sizes(self):
        return self._embedding_sizes

    @property
    def categoricals(self):
        return [*self._static_categoricals, *self._cat_feature]

    @property
    def encoder_cont(self):
        return self._cont_feature

    @property
    def encoder_cat(self):
        return self._cat_feature

    @property
    def max_sequence_length(self):
        return self._indexer.max_sequence_length

    @property
    def lags(self):
        return self._lags

    @property
    def time_features(self):
        return self._time_features

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.
        """
        kwargs = {
            name: getattr(self, f"_{name}")
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["self", "data", "indexer", "last_only", "start_index"]
        }
        kwargs.update(
            indexer=serialize_indexer(self._indexer),
            categorical_encoders=[serialize_encoder(encoder) for encoder in self._categorical_encoders],
            cont_scalars=[serialize_normalizer(scalar) for scalar in self._cont_scalars],
        )
        return kwargs

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any], data: pd.DataFrame, **kwargs):
        parameters = deepcopy(parameters)
        parameters.update(
            indexer=deserialize_indexer(parameters["indexer"]),
            categorical_encoders=[
                deserialize_encoder(encoder) for encoder in parameters["categorical_encoders"]
            ],
            cont_scalars=[deserialize_normalizer(scalar) for scalar in parameters["cont_scalars"]],
            **kwargs,
        )
        return cls(data=data, **parameters)

    @classmethod
    def from_dataset(cls, dataset: "TimeSeriesDataset", data: pd.DataFrame, **kwargs) -> "TimeSeriesDataset":
        return cls.from_parameters(dataset.get_parameters(), data, **kwargs)

    @classmethod
    def from_data_cfg(cls, data_cfg: DataConfig, data: pd.DataFrame, **kwargs):
        indexer = create_indexer_from_cfg(data_cfg.indexer, data_cfg.group_ids)
        embedding_sizes = create_encoder_from_cfg(data_cfg.categoricals)
        return cls(
            data,
            time=data_cfg.time,
            freq=data_cfg.freq,
            group_ids=data_cfg.group_ids,
            indexer=indexer,
            cont_feature=data_cfg.cont_features,
            cat_feature=data_cfg.cat_features,
            # if `data_cfg.categoricals` not provide, regard `cat_features` as no need to process
            categorical_encoders=[PlainEncoder(feature_name=cat) for cat in data_cfg.cat_features]
            if not data_cfg.categoricals
            else [],
            embedding_sizes=embedding_sizes,
            lags=data_cfg.lags,
            **kwargs,
        )

    def __len__(self):
        return len(self._indexer)

    def __getitem__(self, idx: int):
        index = self._indexer[idx]
        encoder_idx = index["steps_idx"]

        encoder_period: pd.DataFrame = self._data.loc[encoder_idx]

        # TODO 缺失值是个值得研究的主题
        encoder_cont = torch.tensor(encoder_period[self.encoder_cont].to_numpy(np.float64), dtype=torch.float)
        encoder_cat = torch.tensor(encoder_period[self.encoder_cat].to_numpy(np.int64), dtype=torch.int)
        encoder_time_idx = encoder_period[TIME_IDX]
        time_idx_start = encoder_time_idx.min()
        encoder_time_idx = torch.tensor(
            (encoder_time_idx - time_idx_start).to_numpy(np.int64), dtype=torch.long
        )

        encoder_target = torch.tensor(encoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)
        time_feature = torch.tensor(
            encoder_period[self.time_features].to_numpy(np.float64), dtype=torch.float
        )
        # inverse `cont` to calculate loss, which is disabled in anomal_detection
        # targets = torch.stack(
        #     [
        #         torch.tensor(
        #             self._cont_scalars[i]
        #             .inverse_transform(encoder_period[target_name], encoder_period)
        #             .to_numpy(np.float64),
        #             dtype=torch.float,
        #         )
        #         for i, target_name in enumerate(self.encoder_cont)
        #     ],
        #     dim=-1,
        # )
        # target_scales = torch.stack(
        #     [
        #         torch.tensor(self._cont_scalars[i].get_norm(encoder_period), dtype=torch.float)
        #         for i, _ in enumerate(self.encoder_cont)
        #     ],
        #     dim=-1,
        # )
        targets = torch.stack(
            [
                torch.tensor(encoder_period[target_name].to_numpy(np.float64), dtype=torch.float)
                for target_name in self.targets
            ],
            dim=-1,
        )
        # [sequence_length, 2, n_targets], concat `cont` and `cat`

        target_scales = (
            torch.cat(
                [
                    torch.stack(
                        [
                            torch.tensor(self._cont_scalars[i].get_norm(encoder_period), dtype=torch.float)
                            for i, _ in enumerate(self.encoder_cont)
                        ],
                        dim=-1,
                    ),
                    torch.stack(
                        [
                            torch.tensor(
                                self._categorical_encoders[i].get_norm(encoder_period), dtype=torch.int
                            )
                            for i, _ in enumerate(self.categoricals)
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )
            if self.categoricals and self.encoder_cont
            else torch.stack(
                [
                    torch.tensor(self._cont_scalars[i].get_norm(encoder_period), dtype=torch.float)
                    for i, _ in enumerate(self.encoder_cont)
                ],
                dim=-1,
            )
            if self.encoder_cont
            else torch.stack(
                [
                    torch.tensor(self._categorical_encoders[i].get_norm(encoder_period), dtype=torch.int)
                    for i, _ in enumerate(self.categoricals)
                ],
                dim=-1,
            )
        )

        return (
            dict(
                encoder_cont=encoder_cont,
                encoder_cat=encoder_cat,
                encoder_time_idx=encoder_time_idx,
                encoder_idx=torch.tensor(encoder_idx, dtype=torch.long),
                encoder_target=encoder_target,
                time_feature=time_feature,
                target_scales=target_scales,
            ),
            targets,
        )

    def _collate_fn(self, batches):
        encoder_cont = torch.stack([batch[0]["encoder_cont"] for batch in batches])
        encoder_cat = torch.stack([batch[0]["encoder_cat"] for batch in batches])
        encoder_time_idx = torch.stack([batch[0]["encoder_time_idx"] for batch in batches])
        encoder_idx = torch.stack([batch[0]["encoder_idx"] for batch in batches])
        encoder_target = torch.stack([batch[0]["encoder_target"] for batch in batches])
        encoder_length = torch.tensor([len(batch[0]["encoder_target"]) for batch in batches])
        time_feature = torch.tensor([len(batch[0]["time_feature"]) for batch in batches])
        target_scales = torch.stack([batch[0]["target_scales"] for batch in batches])
        targets = torch.stack([batch[1] for batch in batches])

        return (
            dict(
                encoder_cont=encoder_cont,
                encoder_cat=encoder_cat,
                encoder_time_idx=encoder_time_idx,
                encoder_idx=encoder_idx,
                encoder_target=encoder_target,
                encoder_length=encoder_length,
                time_feature=time_feature,
                target_scales=target_scales,
            ),
            targets,
        )

    def to_dataloader(self, batch_size: int, train: bool = True, sampler=None, **kwargs) -> DataLoader:
        # np.random.seed(kwargs.get("seed", 46))
        default_kwargs = dict(
            # shuffle=train,
            drop_last=train and len(self) > batch_size,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
        )
        kwargs.update(default_kwargs)
        # return DataLoader(self, **kwargs)

        return DataLoader(
            self, sampler=RandomSampler(self, num_samples=50 * batch_size) if train else None, **kwargs
        )

    def reflect(
        self,
        decoder_indices: List[int],
        inverse_scale_target=True,
        columns: List[str] = None,
    ):
        # group_ids = [
        #     self._categorical_encoders[self.categoricals.index(i)].inverse_transform(i)
        #     for i in self._group_ids
        # ]
        if columns is None:
            if self._time is None:
                columns = [*self.encoder_cont, *self._group_ids]
            else:
                columns = [self._time, *self.encoder_cont, *self._group_ids, "_time_idx_"]

        data_to_return = self._data.loc[[*decoder_indices]][columns]
        # inverse `group_id` column

        # only `cont_features` are considered
        if inverse_scale_target:
            data_to_return = data_to_return.assign(
                **{
                    target_name: self._cont_scalars[i].inverse_transform(
                        data_to_return[target_name], data_to_return
                    )
                    for i, target_name in enumerate(self.encoder_cont)
                }
            )
            data_to_return = data_to_return.assign(
                **{
                    target_name: self._categorical_encoders[i].inverse_transform(data_to_return[target_name])
                    for i, target_name in enumerate(self.encoder_cat)
                }
            )
        if self._group_ids:
            data_to_return = data_to_return.assign(
                **{
                    group: self._categorical_encoders[i].inverse_transform(data_to_return[group])
                    for i, group in enumerate(self.categoricals)
                    if group in self._group_ids
                }
            )
        return data_to_return
