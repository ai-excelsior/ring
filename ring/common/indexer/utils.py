from typing import Tuple
import numpy as np
import pandas as pd


def find_end_indices(
    diffs: pd.Series, max_lengths: np.ndarray, min_length: int, tolerance=np.inf
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify end indices in series even if some values are missing.

    Args:
        diffs (pd.Series): series of differences to next time step. nans should be filled up with ones, we need
        the index, so a pd.series is sent
        max_lengths (np.ndarray): maximum length of sequence by position.
        min_length (int): minimum length of sequence.
        tolerance (int): the max tolerance of missing

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of arrays where first is end indices and second is list of start
            and end indices that are currently missing.
    """
    missing_start_ends = []
    end_indices = []
    length = 1
    start_idx = 0
    max_idx = len(diffs) - 1
    max_length = max_lengths[start_idx]
    tolerance += 1

    # the value of idx is different, but the diffs.index[idx] may be the same due to implementation in predict task
    for idx, diff in enumerate(diffs):
        if length >= max_length:
            while length >= max_length:
                if length == max_length:
                    end_indices.append(diffs.index[idx])
                else:
                    end_indices.append(diffs.index[idx] - 1)
                length -= diffs.iloc[start_idx]
                if start_idx < max_idx:
                    start_idx += 1
                max_length = max_lengths[start_idx]
        elif length >= min_length:
            missing_start_ends.append([start_idx, diffs.index[idx]])

        if diff <= tolerance:
            length += diff
        else:
            # 填充到当前idx的，序列长度为0的截止序列
            end_indices.extend(range(start_idx, diffs.index[idx] + 1))

            # 重置长度、起始位置和该起始位置下的最大长度
            length = 1
            start_idx = diffs.index[idx] + 1
            max_length = max_lengths[start_idx]

    if len(missing_start_ends) > 0:  # required for numba compliance
        return np.asarray(end_indices), np.asarray(missing_start_ends)
    else:
        return np.asarray(end_indices), np.empty((0, 2), dtype=np.int64)


# try:
#     import numba

#     find_end_indices = numba.jit(nopython=True)(find_end_indices)
# except ImportError:
#     pass
