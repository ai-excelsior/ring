import pandas as pd
import numpy as np
import warnings
from typing import List

from .base import BaseIndexer
from .utils import find_end_indices
from ring.common.sample.histogram import ScaleHistogram


class SlideWindowIndexer(BaseIndexer):
    def __init__(
        self,
        time_idx: str,
        look_back: int,
        look_forward: int,
        group_ids: List[str] = [],
    ) -> None:
        self._look_back = look_back
        self._look_forward = look_forward

        self._group_ids = group_ids
        self._time_idx = time_idx

    def index(self, data: pd.DataFrame, predict_mode: bool = False):
        """
        Create index of samples.

        Args:
            data (pd.DataFrame): preprocessed data
            predict_mode (bool): if to create one same per group with prediction length equals ``max_decoder_length``
        """

        if len(self._group_ids) > 0:
            g = data.groupby(self._group_ids, observed=True)
            group_ids = g.ngroup()

            df_time_idx_first = g[self._time_idx].transform("nth", 0).to_frame("time_idx_first")
            df_time_idx_last = g[self._time_idx].transform("nth", -1).to_frame("time_idx_last")
            df_time_idx_to_next = (
                -g[self._time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_idx_to_next")
            )
            df_index = pd.concat([df_time_idx_first, df_time_idx_last, df_time_idx_to_next], axis=1)
            df_index["group_id"] = group_ids
        else:
            df_index = -data[self._time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_idx_to_next")
            df_index["time_idx_first"] = int(data.iloc[0][self._time_idx])
            df_index["time_idx_last"] = int(data.iloc[-1][self._time_idx])

        df_index["index_start"] = np.arange(len(df_index))
        df_index["time_idx"] = data[self._time_idx]

        sequence_length = self._look_back + self._look_forward

        # calculate maximum index to include from current index_start
        max_time_idx = (df_index["time_idx"] + sequence_length - 1).clip(upper=df_index["time_idx_last"])
        df_index["index_end"], missing_sequences = find_end_indices(
            diffs=df_index["time_idx_to_next"].to_numpy(),
            max_lengths=(max_time_idx - df_index["time_idx"]).to_numpy() + 1,
            min_length=sequence_length,
        )

        # add duplicates but mostly with shorter sequence length for start of timeseries
        # while the previous steps have ensured that we start a sequence on every time step, the missing_sequences
        # ensure that there is a sequence that finishes on every timestep
        if len(missing_sequences) > 0:
            shortened_sequences = df_index.iloc[missing_sequences[:, 0]].assign(
                index_end=missing_sequences[:, 1]
            )
            # concatenate shortened sequences
            df_index = pd.concat([df_index, shortened_sequences], axis=0, ignore_index=True)

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = (
            df_index["time_idx"].iloc[df_index["index_end"]].to_numpy() - df_index["time_idx"] + 1
        )

        # filter too short sequences
        # sequence must be at least of minimal prediction length
        df_index = df_index[lambda x: (x["sequence_length"] >= sequence_length)]

        # keep longest element per series (i.e. the first element that spans to the end of the series)
        # filter all elements that are longer than the allowed maximum sequence length
        if predict_mode:
            df_index = df_index[
                lambda x: (x["time_idx_last"] - x["time_idx"] + 1 <= sequence_length)
                & (x["sequence_length"] >= sequence_length)
            ]
            # choose longest sequence
            if len(self._group_ids) > 0:
                df_index = df_index.loc[df_index.groupby("group_id")["sequence_length"].idxmax()]
            else:
                df_index = df_index.loc[[df_index["sequence_length"].idxmax()]]

        # check that all groups/series have at least one entry in the index
        if len(self._group_ids) and not group_ids.isin(df_index["group_id"]).all():
            missing_groups = data.loc[
                ~group_ids.isin(df_index["group_id"]), self._group_ids
            ].drop_duplicates()

            warnings.warn(
                "Min encoder length and/or min_prediction_idx and/or min prediction length and/or lags are "
                "too large for "
                f"{len(missing_groups)} series/groups which therefore are not present in the dataset index. "
                "This means no predictions can be made for those series. ",
                UserWarning,
            )
        assert (
            len(df_index) > 0
        ), "filters should not remove entries all entries - check encoder/decoder lengths and lags"

        self._index = df_index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        """Get an item of this indexer

        Args:
            idx (int): the order of the item

        Returns:
            Dict:
                encoder_idx: encoder part of the indices of the data
                decoder_idx: decoder part of the indices of the data
        """
        index = self._index.iloc[idx]
        index_start = index["index_start"]
        index_end = index["index_end"] + 1

        encoder_length = self._look_back
        decoder_length = self._look_forward

        # TODO randomization encoder length and decoder length to improves generalization
        # return (index_start, index_start + encoder_length, index_end)
        encoder_idx = range(index_start, index_start + encoder_length)
        decoder_idx = range(index_end - decoder_length, index_end)

        return dict(encoder_idx=encoder_idx, decoder_idx=decoder_idx)


class SlideWindowIndexer_fixed(BaseIndexer):
    def __init__(
        self,
        time_idx: str,
        steps: int,
        group_ids: List[str] = [],
    ) -> None:
        self._steps = steps
        self._group_ids = group_ids
        self._time_idx = time_idx

    def index(self, data: pd.DataFrame, last_only: bool = False, start_index: int = None):
        if len(self._group_ids) > 0:
            g = data.groupby(self._group_ids, observed=True)
            group_ids = g.ngroup()

            df_time_idx_first = g[self._time_idx].transform("nth", 0).to_frame("time_idx_first")
            df_time_idx_last = g[self._time_idx].transform("nth", -1).to_frame("time_idx_last")
            df_time_idx_to_next = (
                -g[self._time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_idx_to_next")
            )
            df_index = pd.concat([df_time_idx_first, df_time_idx_last, df_time_idx_to_next], axis=1)
            df_index["group_id"] = group_ids
        else:
            df_index = -data[self._time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_idx_to_next")
            df_index["time_idx_first"] = int(data.iloc[0][self._time_idx])
            df_index["time_idx_last"] = int(data.iloc[-1][self._time_idx])

        df_index["index_start"] = np.arange(len(df_index))
        df_index["time_idx"] = data[self._time_idx]

        sequence_length = self._steps

        # calculate maximum index to include from current index_start
        max_time_idx = (df_index["time_idx"] + sequence_length - 1).clip(upper=df_index["time_idx_last"])
        df_index["index_end"], missing_sequences = find_end_indices(
            diffs=df_index["time_idx_to_next"].to_numpy(),
            max_lengths=(max_time_idx - df_index["time_idx"]).to_numpy() + 1,
            min_length=sequence_length,
        )

        # add duplicates but mostly with shorter sequence length for start of timeseries
        # while the previous steps have ensured that we start a sequence on every time step, the missing_sequences
        # ensure that there is a sequence that finishes on every timestep
        if len(missing_sequences) > 0:
            shortened_sequences = df_index.iloc[missing_sequences[:, 0]].assign(
                index_end=missing_sequences[:, 1]
            )
            # concatenate shortened sequences
            df_index = pd.concat([df_index, shortened_sequences], axis=0, ignore_index=True)

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = (
            df_index["time_idx"].iloc[df_index["index_end"]].to_numpy() - df_index["time_idx"] + 1
        )

        # filter too short sequences
        # sequence must be at least of minimal prediction length
        df_index = df_index[lambda x: (x["sequence_length"] >= sequence_length)]
        if not (len(df_index)):
            raise ValueError("The dataset given is not long enough to meet the steps assigned")

        # only return the last `steps` length result, takes the priority of `start_index`
        if last_only:
            df_index = df_index[
                lambda x: (x["time_idx_last"] - x["time_idx"] + 1 <= sequence_length)
                & (x["sequence_length"] >= sequence_length)
            ]
            # choose longest sequence
            if len(self._group_ids) > 0:
                df_index = df_index.loc[df_index.groupby("group_id")["sequence_length"].idxmax()]
            else:
                df_index = df_index.loc[[df_index["sequence_length"].idxmax()]]

        # specify certain start point
        elif start_index is not None:
            df_index = df_index[
                lambda x: (x["index_start"] >= start_index) & (x["sequence_length"] >= sequence_length)
            ]

            if not (len(df_index)):
                raise ValueError("The start_index given is too large to fetch a batch")

            if len(self._group_ids) > 0:
                df_index = df_index.loc[df_index.groupby("group_id")["sequence_length"]]

            if not (len(df_index)):
                raise ValueError("The start_index given is too large to fetch a batch for each group")

        # check that all groups/series have at least one entry in the index
        if len(self._group_ids) and not group_ids.isin(df_index["group_id"]).all():
            missing_groups = data.loc[
                ~group_ids.isin(df_index["group_id"]), self._group_ids
            ].drop_duplicates()

            warnings.warn(
                "Min encoder length and/or min_prediction_idx and/or min prediction length and/or lags are "
                "too large for "
                f"{len(missing_groups)} series/groups which therefore are not present in the dataset index. "
                "This means no predictions can be made for those series. ",
                UserWarning,
            )
        assert (
            len(df_index) > 0
        ), "filters should not remove entries all entries - check encoder/decoder lengths and lags"

        self._index = df_index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        """Get an item of this indexer

        Args:
            idx (int): the order of the item

        Returns:
            Dict:
                encoder_idx: encoder part of the indices of the data
                decoder_idx: decoder part of the indices of the data
        """
        index = self._index.iloc[idx]
        index_start = index["index_start"]
        index_end = index["index_end"] + 1

        # TODO randomization encoder length and decoder length to improves generalization
        # return (index_start, index_start + encoder_length, index_end)
        steps_idx = range(index_start, index_end)

        return dict(steps_idx=steps_idx)


class SlideWindowIndexer_bucketSampler(BaseIndexer):
    def __init__(
        self,
        time_idx: str,
        look_back: int,
        look_forward: int,
        group_ids: List[str] = [],
    ) -> None:
        self._look_back = look_back
        self._look_forward = look_forward

        self._group_ids = group_ids
        self._time_idx = time_idx
        self.scale_histogram = ScaleHistogram()

    def index(self, data: pd.DataFrame, predict_mode: bool = False):
        """
        Create index of samples.

        Args:
            data (pd.DataFrame): preprocessed data
            predict_mode (bool): if to create one same per group with prediction length equals ``max_decoder_length``
        """

        if len(self._group_ids) > 0:
            g = data.groupby(self._group_ids, observed=True)
            group_ids = g.ngroup()

            df_time_idx_first = g[self._time_idx].transform("nth", 0).to_frame("time_idx_first")
            df_time_idx_last = g[self._time_idx].transform("nth", -1).to_frame("time_idx_last")
            df_time_idx_to_next = (
                -g[self._time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_idx_to_next")
            )
            df_index = pd.concat([df_time_idx_first, df_time_idx_last, df_time_idx_to_next], axis=1)
            df_index["group_id"] = group_ids

        else:
            df_index = -data[self._time_idx].diff(-1).fillna(-1).astype(int).to_frame("time_idx_to_next")
            df_index["time_idx_first"] = int(data.iloc[0][self._time_idx])
            df_index["time_idx_last"] = int(data.iloc[-1][self._time_idx])

        df_index["index_start"] = np.arange(len(df_index))
        df_index["time_idx"] = data[self._time_idx]

        sequence_length = self._look_back + self._look_forward

        # calculate maximum index to include from current index_start
        max_time_idx = (df_index["time_idx"] + sequence_length - 1).clip(upper=df_index["time_idx_last"])
        df_index["index_end"], missing_sequences = find_end_indices(
            diffs=df_index["time_idx_to_next"].to_numpy(),
            max_lengths=(max_time_idx - df_index["time_idx"]).to_numpy() + 1,
            min_length=sequence_length,
        )

        # add duplicates but mostly with shorter sequence length for start of timeseries
        # while the previous steps have ensured that we start a sequence on every time step, the missing_sequences
        # ensure that there is a sequence that finishes on every timestep
        if len(missing_sequences) > 0:
            shortened_sequences = df_index.iloc[missing_sequences[:, 0]].assign(
                index_end=missing_sequences[:, 1]
            )
            # concatenate shortened sequences
            df_index = pd.concat([df_index, shortened_sequences], axis=0, ignore_index=True)

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = (
            df_index["time_idx"].iloc[df_index["index_end"]].to_numpy() - df_index["time_idx"] + 1
        )

        # filter too short sequences
        # sequence must be at least of minimal prediction length
        df_index = df_index[lambda x: (x["sequence_length"] >= sequence_length)]

        # keep longest element per series (i.e. the first element that spans to the end of the series)
        # filter all elements that are longer than the allowed maximum sequence length
        if predict_mode:
            df_index = df_index[
                lambda x: (x["time_idx_last"] - x["time_idx"] + 1 <= sequence_length)
                & (x["sequence_length"] >= sequence_length)
            ]
            # choose longest sequence
            if len(self._group_ids) > 0:
                df_index = df_index.loc[df_index.groupby("group_id")["sequence_length"].idxmax()]
            else:
                df_index = df_index.loc[[df_index["sequence_length"].idxmax()]]

        # check that all groups/series have at least one entry in the index
        if len(self._group_ids) and not group_ids.isin(df_index["group_id"]).all():
            missing_groups = data.loc[
                ~group_ids.isin(df_index["group_id"]), self._group_ids
            ].drop_duplicates()

            warnings.warn(
                "Min encoder length and/or min_prediction_idx and/or min prediction length and/or lags are "
                "too large for "
                f"{len(missing_groups)} series/groups which therefore are not present in the dataset index. "
                "This means no predictions can be made for those series. ",
                UserWarning,
            )
        assert (
            len(df_index) > 0
        ), "filters should not remove entries all entries - check encoder/decoder lengths and lags"

        if len(self._group_ids) > 0:
            # 计算每个group的取样概率
            unique_group_ids = data[self._group_ids[0]].unique()
            sequence_length = self._look_back + self._look_forward
            sample_df_index = []
            for i in unique_group_ids:
                group_data = data[data[self._group_ids[0]] == i]
                group_target = group_data["sales"]
                self.scale_histogram.add(group_target.values)

            for i in unique_group_ids:
                group_df_index = df_index[df_index["group_id"] == i]
                group_data = data[data[self._group_ids[0]] == i]
                group_target = group_data["sales"]
                p_i = 1.0 / self.scale_histogram.count(group_target.values)
                (indices,) = np.where(np.random.random_sample(len(group_df_index)) < p_i)
                if len(indices) > 0:
                    sample_group_i = group_df_index.loc[group_df_index["time_idx"].isin(indices)]
                    sample_df_index.append(sample_group_i)
            df_index = pd.concat(sample_df_index)
            # print(len(df_index))

        self._index = df_index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        """Get an item of this indexer

        Args:
            idx (int): the order of the item

        Returns:
            Dict:
                encoder_idx: encoder part of the indices of the data
                decoder_idx: decoder part of the indices of the data
        """
        index = self._index.iloc[idx]
        index_start = index["index_start"]
        index_end = index["index_end"] + 1

        encoder_length = self._look_back
        decoder_length = self._look_forward

        # TODO randomization encoder length and decoder length to improves generalization
        # return (index_start, index_start + encoder_length, index_end)
        encoder_idx = range(index_start, index_start + encoder_length)
        decoder_idx = range(index_end - decoder_length, index_end)

        return dict(encoder_idx=encoder_idx, decoder_idx=decoder_idx)
