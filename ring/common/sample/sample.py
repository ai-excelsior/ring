from torch.utils.data.sampler import Sampler
from typing import Iterator, Optional, Sized
import torch
import numpy as np
import pandas as pd


class ExpectedNumInstanceSampler(Sampler[int]):
    """
    Keeps track of the average time series length and adjusts the probability
    per time point such that on average `num_instances` training examples are
    generated per time series.
    Parameters
    ----------
    num_instances
        number of training examples generated per time series on average
    """

    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value,but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        else:
            if len(self.data_source._group_ids) > 0:
                index_ts = self.data_source._indexer._index
                unique_group_id = index_ts["group_id"].unique()
                num_samples_all = 0

                for i in unique_group_id:
                    df_for_group_i = index_ts[index_ts["group_id"] == i]
                    length_for_group_i = len(df_for_group_i)
                    sample_num_group_i = max(
                        1, int(self._num_samples * (length_for_group_i / len(self.data_source)))
                    )
                    num_samples_all += sample_num_group_i
                return num_samples_all
            else:
                return self._num_samples

    def __iter__(self) -> Iterator[int]:

        length_data_source = len(self.data_source)
        sample_group_all = []
        sample_data = pd.DataFrame(columns=self.data_source._indexer._index.columns)

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if len(self.data_source._group_ids) > 0:
            index_ts = self.data_source._indexer._index
            index_ts = index_ts.reset_index(drop=True)
            unique_group_id = index_ts["group_id"].unique()

            for i in unique_group_id:
                df_for_group_i = index_ts[index_ts["group_id"] == i]
                length_for_group_i = len(df_for_group_i)
                sample_num_group_i = max(
                    1, int(self._num_samples * (length_for_group_i / length_data_source))
                )

                L = df_for_group_i["time_idx"].min()
                H = df_for_group_i["time_idx"].max()
                indices = []

                while len(indices) < sample_num_group_i:
                    indice = np.array(torch.randint(L, H, (1,)).flatten())[0]
                    if indice not in np.unique(indices):
                        indices.append(indice)

                sample_group_i = df_for_group_i.loc[df_for_group_i["time_idx"].isin(indices)]
                sample_group_all.append(sample_group_i)

            sample_data = pd.concat(sample_group_all)
            yield from sample_data.index.values.tolist()

        else:
            for _ in range(self.num_samples // length_data_source):
                yield from torch.randperm(length_data_source, generator=generator).tolist()
            yield from torch.randperm(length_data_source, generator=generator).tolist()[
                : self.num_samples % length_data_source
            ]

    def __len__(self) -> int:
        return self.num_samples
