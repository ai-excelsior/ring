import pandas as pd
import numpy as np
import torch
import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from ring.common.data_config import DataConfig
from ring.common.time_features import time_feature
from .indexer import BaseIndexer, create_indexer_from_cfg, serialize_indexer, deserialize_indexer
from .normalizers import (
    AbstractNormalizer,
    GroupStardardNormalizer,
    Normalizer,
    StandardNormalizer,
    serialize_normalizer,
    deserialize_normalizer,
)
from .encoder import LabelEncoder, deserialize_encoder, serialize_encoder
from .seasonality import (
    serialize_detrender,
    deserialize_detrender,
    serialize_lags,
    deserialize_lags,
    create_detrender_from_cfg,
    create_lags_from_cfg,
)
from .utils import get_default_embedding_size, add_time_idx
from .encoder import create_encoder_from_cfg
from torch.utils.data.sampler import RandomSampler

STATIC_UNKNOWN_REAL_NAME = "_ZERO_"
TIME_IDX = "_time_idx_"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        time: str,
        freq: str,
        indexer: BaseIndexer,
        targets: List[str],
        group_ids: List[str] = [],
        # 支持的协变量们
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
        embedding_sizes: Dict[str, Tuple[str, str]] = None,
        time_features: List = [],
        # normalizers
        target_normalizers: List[Normalizer] = [],
        categorical_encoders: List[LabelEncoder] = [],
        cont_scalars: List[StandardNormalizer] = [],
        target_detrenders: Union[callable, None] = None,
        # toggles
        predict_mode=False,  # using predict mode to index
        enable_static_as_covariant=True,
        add_static_known_real=None,  # add zero to time varying known (decoder part) field
        # seasonality
        lags: Union[Dict, None] = None,
        # need detrend or not
    ) -> None:
        self._data = data
        self._time = time
        self._freq = freq
        self._group_ids = group_ids
        self._indexer = indexer
        self._targets = targets
        self._static_categoricals = static_categoricals
        self._static_reals = static_reals
        self._time_varying_known_categoricals = time_varying_known_categoricals
        self._time_varying_known_reals = time_varying_known_reals
        self._time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self._time_varying_unknown_reals = time_varying_unknown_reals
        self._embedding_sizes = embedding_sizes or {}
        self._enable_static_as_covariant = enable_static_as_covariant
        self._add_static_known_real = add_static_known_real
        self._target_normalizers = target_normalizers
        self._categorical_encoders = categorical_encoders
        self._cont_scalars = cont_scalars
        self._target_detrenders = target_detrenders
        self._lags = lags
        self._known_lag_features = []
        self._unknown_lag_features = []

        # `time_features` will not be added in `decoder_cont` or `encoder_cont`, they will be processed in model
        # train/val/pred: do not add `time_features`
        if time_features is None or (not time_features and predict_mode):
            self._time_features = []
        # train/val/pred: `time_features` already all in data
        elif time_features and not sum([item not in data.columns for item in time_features]):
            self._time_features = time_features
            # make sure time_features always at the end of dataset
            data = data[[c for c in data.columns if c not in self._time_features] + self._time_features]
        # train/val/pred: `time_features` added automatically by code
        elif isinstance(time_features, list):
            time_feature_data = time_feature(pd.to_datetime(data[self._time].values), freq=self._freq)
            self._time_features = list(time_feature_data.columns)
            # make sure time_features always at the end of dataset
            data = pd.concat([data, time_feature_data], axis=1)

        # create target normalizer
        if len(self._target_normalizers) == 0:
            for tar in self.targets:
                if len(self._group_ids) > 0:
                    self._target_normalizers.append(
                        GroupStardardNormalizer(group_ids=self._group_ids, feature_name=tar)
                    )
                else:
                    self._target_normalizers.append(StandardNormalizer(feature_name=tar))

        # create categorical encoder
        if len(self._categorical_encoders) == 0:
            for cat in self.categoricals:
                self._categorical_encoders.append(LabelEncoder(feature_name=cat))

        # create continouse scalar
        if len(self._cont_scalars) == 0:
            for i, cont in enumerate(set(self.encoder_cont) - set(self.targets)):
                if len(self._group_ids) > 0:
                    self._cont_scalars.append(
                        GroupStardardNormalizer(group_ids=self._group_ids, feature_name=cont)
                    )
                else:
                    self._cont_scalars.append(StandardNormalizer(feature_name=cont))

        assert all(
            group_id in data.columns for group_id in group_ids
        ), f"Please make sure all {group_ids} is in the data"

        # create embedding_size
        for categorical_name in self.categoricals:
            if categorical_name not in self._embedding_sizes:
                cat_size = len(data[categorical_name].unique())
                self._embedding_sizes[categorical_name] = (cat_size + 1, get_default_embedding_size(cat_size))
                # with unknow capacity

        # initialize indexer
        data = add_time_idx(data, time_column_name=time, freq=freq)
        data.sort_values([*self._group_ids, TIME_IDX], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # detrend targets, all targets together at once
        if not self._target_detrenders.fitted:
            data[self.targets] = self._target_detrenders.fit_transform(data, self._group_ids)
        else:
            data[self.targets] = self._target_detrenders.transform(data, self._group_ids)

        # fit categoricals
        for i, cat in enumerate(self.categoricals):
            encoder = self._categorical_encoders[i]
            if not encoder.fitted:
                data[cat] = encoder.fit_transform(data[cat].astype(str))

            else:
                data[cat] = encoder.transform(data[cat].astype(str), self.embedding_sizes)

        # fit target normalizers
        for i, target_name in enumerate(self._targets):
            normalizer = self._target_normalizers[i]
            if not normalizer.fitted:
                data[target_name] = normalizer.fit_transform(data[target_name], data)
            else:
                data[target_name] = normalizer.transform(data[target_name], data)

        # fit continous scalar
        for i, cont in enumerate(set(self.encoder_cont) - set(self.targets)):
            scalar = self._cont_scalars[i]
            if not scalar.fitted:
                data[cont] = scalar.fit_transform(data[cont], data)
            else:
                data[cont] = scalar.transform(data[cont], data)

        # obtain target lags through seasonality
        if self._lags:
            for tar in self._targets:
                data = pd.concat(
                    [
                        data,
                        self._lags[tar].add_lags(
                            data[["_time_idx_", tar] + self._group_ids], self._group_ids
                        ),
                    ],
                    axis=1,
                )
                self._known_lag_features += [
                    k for k, v in self._lags[tar]._state.items() if v >= self._indexer._look_forward
                ]
                self._unknown_lag_features += [
                    k for k, v in self._lags[tar]._state.items() if v < self._indexer._look_forward
                ]

        # commonly, `time_features` have been calculated, so `__ZERO__` will not be added
        # but some frequency has no time_features, e.g. Year, or `time_features` should not be added
        if self._add_static_known_real is None and (len(self.decoder_cont) + len(self.decoder_cat)) == 0:
            self._add_static_known_real = True
        if self._add_static_known_real is True:
            data[STATIC_UNKNOWN_REAL_NAME] = 0.0

        # remove nan caused by lag shift, apply index
        self._data = data.dropna().reset_index(drop=True)
        self._indexer.index(self._data, predict_mode)

    @property
    def targets(self):
        return self._targets

    @property
    def n_targets(self):
        return len(self._targets)

    @property
    def target_idx(self):
        return self.encoder_cont.index(self._targets)

    @property
    def target_normalizers(self) -> List[AbstractNormalizer]:
        return self._target_normalizers

    @property
    def embedding_sizes(self):
        return self._embedding_sizes

    @property
    def categoricals(self):
        return [
            *self._group_ids,
            *self._static_categoricals,
            *self._time_varying_known_categoricals,
            *self._time_varying_unknown_categoricals,
        ]

    @property
    def encoder_cont(self):
        if self._enable_static_as_covariant:
            return [
                *self._time_varying_known_reals,
                *self._time_varying_unknown_reals,
                *self._static_reals,
                *self.targets,
            ]

        return [
            *self._time_varying_known_reals,
            *self._time_varying_unknown_reals,
            *self.targets,
        ]

    @property
    def encoder_cat(self):
        if self._enable_static_as_covariant:
            return [
                *self._time_varying_known_categoricals,
                *self._time_varying_unknown_categoricals,
                *self._group_ids,
                *self._static_categoricals,
            ]

        return [*self._time_varying_known_categoricals, *self._time_varying_unknown_categoricals]

    @property
    def decoder_cont(self):
        if self._enable_static_as_covariant:
            return [
                *self._time_varying_known_reals,
                *self._static_reals,
                *([STATIC_UNKNOWN_REAL_NAME] if self._add_static_known_real is True else []),
            ]

        return [
            *self._time_varying_known_reals,
            *([STATIC_UNKNOWN_REAL_NAME] if self._add_static_known_real is True else []),
        ]

    @property
    def decoder_cat(self):
        if self._enable_static_as_covariant:
            return [
                *self._time_varying_known_categoricals,
                *self._group_ids,
                *self._static_categoricals,
            ]

        return [*self._time_varying_known_categoricals]

    @property
    def max_sequence_length(self):
        return self._indexer.max_sequence_length

    @property
    def lags(self):
        return [v for _, v in self._lags.items()]

    @property
    def time_features(self):
        return self._time_features

    @property
    def encoder_lag_features(self):
        return [*self._known_lag_features, *self._unknown_lag_features]

    @property
    def decoder_lag_features(self):
        return [*self._known_lag_features]

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.
        """
        kwargs = {
            name: getattr(self, f"_{name}")
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["self", "data", "indexer", "target_normalizer", "predict_mode"]
        }
        kwargs.update(
            indexer=serialize_indexer(self._indexer),
            target_normalizers=[serialize_normalizer(normalizer) for normalizer in self._target_normalizers],
            categorical_encoders=[serialize_encoder(encoder) for encoder in self._categorical_encoders],
            cont_scalars=[serialize_normalizer(scalar) for scalar in self._cont_scalars],
            target_detrenders=serialize_detrender(self._target_detrenders),
            lags=[serialize_lags(lag) for lag in self.lags],
        )
        return kwargs

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any], data: pd.DataFrame, **kwargs):
        parameters = deepcopy(parameters)
        parameters.update(
            indexer=deserialize_indexer(parameters["indexer"]),
            target_normalizers=[
                deserialize_normalizer(params) for params in parameters["target_normalizers"]
            ],
            categorical_encoders=[
                deserialize_encoder(encoder) for encoder in parameters["categorical_encoders"]
            ],
            cont_scalars=[deserialize_normalizer(scalar) for scalar in parameters["cont_scalars"]],
            target_detrenders=deserialize_detrender(parameters["target_detrenders"]),
            lags=deserialize_lags(parameters["lags"]),
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
        target_detrenders = create_detrender_from_cfg(
            data_cfg.lags is None, data_cfg.detrend, data_cfg.group_ids, data_cfg.targets
        )
        lags = create_lags_from_cfg(data_cfg.lags, data_cfg.group_ids, data_cfg.targets)

        return cls(
            data,
            time=data_cfg.time,
            freq=data_cfg.freq,
            targets=data_cfg.targets,
            group_ids=data_cfg.group_ids,
            indexer=indexer,
            static_categoricals=data_cfg.static_categoricals,
            static_reals=data_cfg.static_reals,
            time_varying_known_categoricals=data_cfg.time_varying_known_categoricals,
            time_varying_known_reals=data_cfg.time_varying_known_reals,
            time_varying_unknown_categoricals=data_cfg.time_varying_unknown_categoricals,
            time_varying_unknown_reals=data_cfg.time_varying_unknown_reals,
            embedding_sizes=embedding_sizes,
            time_features=data_cfg.time_features,
            lags=lags,
            target_detrenders=target_detrenders,
            **kwargs,
        )

    def __len__(self):
        return len(self._indexer)

    def __getitem__(self, idx: int):
        index = self._indexer[idx]
        encoder_idx = index["encoder_idx"]
        decoder_idx = index["decoder_idx"]
        encoder_period: pd.DataFrame = self._data.loc[encoder_idx]
        decoder_period: pd.DataFrame = self._data.loc[decoder_idx]

        # TODO 缺失值是个值得研究的主题
        encoder_cont = torch.tensor(encoder_period[self.encoder_cont].to_numpy(np.float64), dtype=torch.float)
        encoder_cat = torch.tensor(encoder_period[self.encoder_cat].to_numpy(np.int64), dtype=torch.int)
        encoder_time_idx = encoder_period[TIME_IDX]
        time_idx_start = encoder_time_idx.min()
        encoder_time_idx = torch.tensor(
            (encoder_time_idx - time_idx_start).to_numpy(np.int64), dtype=torch.long
        )
        encoder_target = torch.tensor(encoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)
        encoder_time_features = torch.tensor(
            encoder_period[self.time_features].to_numpy(np.float64), dtype=torch.float
        )
        encoder_lag_features = torch.tensor(
            encoder_period[self.encoder_lag_features].to_numpy(np.float64),
            dtype=torch.float,
        )

        decoder_cont = torch.tensor(decoder_period[self.decoder_cont].to_numpy(np.float64), dtype=torch.float)
        decoder_cat = torch.tensor(decoder_period[self.decoder_cat].to_numpy(np.float64), dtype=torch.int)
        decoder_time_idx = decoder_period[TIME_IDX]
        decoder_time_idx = torch.tensor(
            (decoder_time_idx - time_idx_start).to_numpy(np.int64), dtype=torch.long
        )

        decoder_target = torch.tensor(decoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)
        decoder_time_features = torch.tensor(
            decoder_period[self.time_features].to_numpy(np.float64), dtype=torch.float
        )
        decoder_lag_features = torch.tensor(
            decoder_period[self.decoder_lag_features].to_numpy(np.float64),
            dtype=torch.float,
        )
        targets = torch.tensor(decoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)
        # [sequence_length, n_targets]
        # targets = torch.stack(
        #     [
        #         torch.tensor(
        #             self._target_normalizers[i]
        #             .inverse_transform(decoder_period[target_name], decoder_period)
        #             .to_numpy(np.float64),
        #             dtype=torch.float,
        #         )
        #         for i, target_name in enumerate(self.targets)
        #     ],
        #     dim=-1,
        # )
        # targets_back = torch.stack(
        #     [
        #         torch.tensor(
        #             self._target_normalizers[i]
        #             .inverse_transform(encoder_period[target_name], encoder_period)
        #             .to_numpy(np.float64),
        #             dtype=torch.float,
        #         )
        #         for i, target_name in enumerate(self.targets)
        #     ],
        #     dim=-1,
        # )

        # [sequence_length, 2, n_targets]
        target_scales = torch.stack(
            [
                torch.tensor(self._target_normalizers[i].get_norm(decoder_period), dtype=torch.float)
                for i, _ in enumerate(self.targets)
            ],
            dim=-1,
        )
        target_scales_back = torch.stack(
            [
                torch.tensor(self._target_normalizers[i].get_norm(encoder_period), dtype=torch.float)
                for i, _ in enumerate(self.targets)
            ],
            dim=-1,
        )

        return (
            dict(  # batch[0]
                encoder_cont=encoder_cont,
                encoder_cat=encoder_cat,
                encoder_time_idx=encoder_time_idx,
                encoder_idx=torch.tensor(encoder_idx, dtype=torch.long),
                encoder_target=encoder_target,
                encoder_time_features=encoder_time_features,
                encoder_lag_features=encoder_lag_features,
                decoder_cont=decoder_cont,
                decoder_cat=decoder_cat,
                decoder_time_idx=decoder_time_idx,
                decoder_idx=torch.tensor(decoder_idx, dtype=torch.long),
                decoder_target=decoder_target,
                decoder_time_features=decoder_time_features,
                decoder_lag_features=decoder_lag_features,
                target_scales=target_scales,
                target_scales_back=target_scales_back,
            ),
            targets,  # batch[1]
        )

    def _collate_fn(self, batches):
        encoder_cont = torch.stack([batch[0]["encoder_cont"] for batch in batches])
        encoder_cat = torch.stack([batch[0]["encoder_cat"] for batch in batches])
        encoder_time_idx = torch.stack([batch[0]["encoder_time_idx"] for batch in batches])
        encoder_idx = torch.stack([batch[0]["encoder_idx"] for batch in batches])
        encoder_target = torch.stack([batch[0]["encoder_target"] for batch in batches])
        encoder_length = torch.tensor([len(batch[0]["encoder_target"]) for batch in batches])

        decoder_cont = torch.stack([batch[0]["decoder_cont"] for batch in batches])
        decoder_cat = torch.stack([batch[0]["decoder_cat"] for batch in batches])
        decoder_time_idx = torch.stack([batch[0]["decoder_time_idx"] for batch in batches])
        decoder_idx = torch.stack([batch[0]["decoder_idx"] for batch in batches])
        decoder_target = torch.stack([batch[0]["decoder_target"] for batch in batches])
        decoder_length = torch.tensor([len(batch[0]["decoder_target"]) for batch in batches])

        target_scales = torch.stack([batch[0]["target_scales"] for batch in batches])
        target_scales_back = torch.stack([batch[0]["target_scales_back"] for batch in batches])
        targets = torch.stack([batch[1] for batch in batches])
        encoder_time_features = torch.stack([batch[0]["encoder_time_features"] for batch in batches])
        decoder_time_features = torch.stack([batch[0]["decoder_time_features"] for batch in batches])
        encoder_lag_features = torch.stack([batch[0]["encoder_lag_features"] for batch in batches])
        decoder_lag_features = torch.stack([batch[0]["decoder_lag_features"] for batch in batches])
        return (
            dict(
                encoder_cont=encoder_cont,
                encoder_cat=encoder_cat,
                encoder_time_idx=encoder_time_idx,
                encoder_idx=encoder_idx,
                encoder_target=encoder_target,
                encoder_length=encoder_length,
                decoder_cont=decoder_cont,
                decoder_cat=decoder_cat,
                decoder_time_idx=decoder_time_idx,
                decoder_idx=decoder_idx,
                decoder_target=decoder_target,
                decoder_length=decoder_length,
                target_scales=target_scales,
                target_scales_back=target_scales_back,
                encoder_time_features=encoder_time_features,
                decoder_time_features=decoder_time_features,
                encoder_lag_features=encoder_lag_features,
                decoder_lag_features=decoder_lag_features,
            ),
            targets,
        )

    def to_dataloader(self, batch_size: int, train: bool = True, **kwargs) -> DataLoader:
        default_kwargs = dict(
            # shuffle=train,
            drop_last=train and len(self) > batch_size,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
        )
        kwargs.update(default_kwargs)

        return DataLoader(
            self, sampler=RandomSampler(self, num_samples=50 * batch_size) if train else None, **kwargs
        )

    def reflect(
        self,
        encoder_indices: List[int],
        decoder_indices: List[int],
        inverse_scale_target=True,
        columns: List[str] = None,
    ):
        if columns is None:
            if self._time is None:
                columns = [*self._group_ids, *self._targets]
            else:
                columns = [self._time, *self._group_ids, *self._targets, "_time_idx_"]
        # get desired part
        data_to_return = self._data.loc[[*encoder_indices, *decoder_indices]][columns]

        # add decoder part is_prediction is True
        data_to_return = data_to_return.assign(is_prediction=False)
        data_to_return.loc[decoder_indices, "is_prediction"] = True

        # inverse normalize
        if inverse_scale_target:
            data_to_return = data_to_return.assign(
                **{
                    target_name: self._target_normalizers[i].inverse_transform(
                        data_to_return[target_name], data_to_return
                    )
                    for i, target_name in enumerate(self._targets)
                }
            )
        if self._group_ids:
            data_to_return = data_to_return.assign(
                **{
                    group_id: self._categorical_encoders[i].inverse_transform(data_to_return[group_id])
                    for i, group_id in enumerate(self.categoricals)
                    if group_id in self._group_ids
                }
            )
        # inverse detrend
        data_to_return[self.targets] = self._target_detrenders.inverse_transform(
            data_to_return, self._group_ids
        )

        return data_to_return
