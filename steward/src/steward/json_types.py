"""NumPy containers/scalars at JSON boundaries; never stringify unknown objects."""
from typing import Any

import numpy as np


def numpy_json(value: Any) -> Any:
    """JSON encoder default. Non-finite floats still obey the encoder's allow_nan=False."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.generic):
        item = value.item()
        if not isinstance(item, np.generic):
            return item
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def plain_json(value: Any) -> Any:
    """Copy a nested receipt into Python containers and ordinary Python scalars."""
    if isinstance(value, (np.ndarray, np.generic)):
        return plain_json(numpy_json(value))
    if isinstance(value, dict):
        return {plain_json(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    return value
