import pandas as pd
import numpy as np
import torch
import inspect
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader

from ring.common.data_config import DataConfig
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
        # normalizers
        target_normalizers: List[Normalizer] = [],
        categorical_encoders: List[LabelEncoder] = [],
        cont_scalars: List[StandardNormalizer] = [],
        # toggles
        predict_mode=False,  # using predict mode to index
        enable_static_as_covariant=True,
        add_static_known_real=None,  # add zero to time varying known (decoder part) field
        # seasonality
        lags: Dict = {},
    ) -> None:
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
        self._lags = lags
        # target normalizer
        if len(self._target_normalizers) == 0:
            for _ in self.targets:
                if len(self._group_ids) > 0:
                    self._target_normalizers.append(
                        GroupStardardNormalizer(group_ids=self._group_ids, feature_name=_)
                    )
                else:
                    self._target_normalizers.append(StandardNormalizer(feature_name=_))

        # categorical encoder
        if len(self._categorical_encoders) == 0:
            for cat in self.categoricals:
                self._categorical_encoders.append(LabelEncoder(feature_name=cat))

        # continouse scalar
        if len(self._cont_scalars) == 0:
            for i, cont in enumerate(set(self.encoder_cont) - set(self.targets)):
                if len(self._group_ids) > 0:
                    self._cont_scalars.append(
                        GroupStardardNormalizer(group_ids=self._group_ids, feature_name=cont)
                    )
                else:
                    self._cont_scalars.append(StandardNormalizer(feature_name=cont))

        # 确保group_ids存在
        assert all(
            group_id in data.columns for group_id in group_ids
        ), f"Please make sure all {group_ids} is in the data"

        # 对于所有categoricals值，创建默认的embedding_size
        for categorical_name in self.categoricals:
            if categorical_name not in self._embedding_sizes:
                cat_size = len(data[categorical_name].unique())
                self._embedding_sizes[categorical_name] = (cat_size + 1, get_default_embedding_size(cat_size))
                # with unknow capacity

        # 初始化indexer
        data = add_time_idx(data, time_column_name=time, freq=freq)
        data.sort_values([*self._group_ids, TIME_IDX], inplace=True)
        data.reset_index(drop=True, inplace=True)
        self._indexer.index(data, predict_mode)

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

        self._data = data

        if self._add_static_known_real is None and (len(self.decoder_cont) + len(self.decoder_cat)) == 0:
            self._add_static_known_real = True
        if self._add_static_known_real is True:
            data[STATIC_UNKNOWN_REAL_NAME] = 0.0

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
                *self._static_categoricals,
            ]

        return [*self._time_varying_known_categoricals, *self._time_varying_unknown_categoricals]

    @property
    def decoder_cont(self):
        if self._enable_static_as_covariant:
            return [
                *self._time_varying_known_reals,
                *self._static_reals,
                *self.targets,
                *([STATIC_UNKNOWN_REAL_NAME] if self._add_static_known_real is True else []),
            ]

        return [
            *self._time_varying_known_reals,
            *self.targets,
            *([STATIC_UNKNOWN_REAL_NAME] if self._add_static_known_real is True else []),
        ]

    @property
    def decoder_cat(self):
        if self._enable_static_as_covariant:
            return [
                *self._time_varying_known_categoricals,
                *self._static_categoricals,
            ]

        return [*self._time_varying_known_categoricals]

    @property
    def max_sequence_length(self):
        return self._indexer.max_sequence_length

    @property
    def lags(self):
        return self._lags

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
            lags=data_cfg.lags,
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

        decoder_cont = torch.tensor(decoder_period[self.decoder_cont].to_numpy(np.float64), dtype=torch.float)
        decoder_cat = torch.tensor(decoder_period[self.decoder_cat].to_numpy(np.float64), dtype=torch.int)
        decoder_time_idx = decoder_period[TIME_IDX]
        decoder_time_idx = torch.tensor(
            (decoder_time_idx - time_idx_start).to_numpy(np.int64), dtype=torch.long
        )

        decoder_target = torch.tensor(decoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)

        # [sequence_length, n_targets]
        targets = torch.stack(
            [
                torch.tensor(
                    self._target_normalizers[i]
                    .inverse_transform(decoder_period[target_name], decoder_period)
                    .to_numpy(np.float64),
                    dtype=torch.float,
                )
                for i, target_name in enumerate(self.targets)
            ],
            dim=-1,
        )
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
            dict(
                encoder_cont=encoder_cont,
                encoder_cat=encoder_cat,
                encoder_time_idx=encoder_time_idx,
                encoder_idx=torch.tensor(encoder_idx, dtype=torch.long),
                encoder_target=encoder_target,
                decoder_cont=decoder_cont,
                decoder_cat=decoder_cat,
                decoder_time_idx=decoder_time_idx,
                decoder_idx=torch.tensor(decoder_idx, dtype=torch.long),
                decoder_target=decoder_target,
                target_scales=target_scales,
                target_scales_back=target_scales_back,
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

        decoder_cont = torch.stack([batch[0]["decoder_cont"] for batch in batches])
        decoder_cat = torch.stack([batch[0]["decoder_cat"] for batch in batches])
        decoder_time_idx = torch.stack([batch[0]["decoder_time_idx"] for batch in batches])
        decoder_idx = torch.stack([batch[0]["decoder_idx"] for batch in batches])
        decoder_target = torch.stack([batch[0]["decoder_target"] for batch in batches])
        decoder_length = torch.tensor([len(batch[0]["decoder_target"]) for batch in batches])

        target_scales = torch.stack([batch[0]["target_scales"] for batch in batches])
        target_scales_back = torch.stack([batch[0]["target_scales_back"] for batch in batches])
        targets = torch.stack([batch[1] for batch in batches])

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
            ),
            targets,
        )

    def to_dataloader(self, batch_size: int, train: bool = True, **kwargs) -> DataLoader:
        default_kwargs = dict(
            shuffle=train,
            drop_last=train and len(self) > batch_size,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
        )
        kwargs.update(default_kwargs)
        return DataLoader(self, **kwargs)

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

        data_to_return = self._data.loc[[*encoder_indices, *decoder_indices]][columns]

        # add decoder part is_prediction is True
        data_to_return = data_to_return.assign(is_prediction=False)
        data_to_return.loc[decoder_indices, "is_prediction"] = True

        if inverse_scale_target:
            data_to_return = data_to_return.assign(
                **{
                    target_name: self._target_normalizers[i].inverse_transform(
                        self._data[target_name], self._data
                    )
                    for i, target_name in enumerate(self._targets)
                }
            )

        return data_to_return
