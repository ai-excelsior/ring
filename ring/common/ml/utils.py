from typing import Any, List
from torch.nn.utils import rnn


def to_list(value: Any) -> List[Any]:
    """
    Convert value or list to list of values.
    If already list, return object directly

    Args:
        value (Any): value to convert

    Returns:
        List[Any]: list of values
    """
    if isinstance(value, (tuple, list)) and not isinstance(value, rnn.PackedSequence):
        return value
    else:
        return [value]
