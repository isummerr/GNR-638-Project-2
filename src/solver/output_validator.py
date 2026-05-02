from __future__ import annotations
import math
from typing import Any
ALLOWED = frozenset({1, 2, 3, 4, 5})
ABSTAIN = 5

def coerce_option(raw: Any) -> int:
    if isinstance(raw, bool):
        return ABSTAIN
    if raw is None:
        return ABSTAIN
    if isinstance(raw, float):
        if math.isnan(raw) or math.isinf(raw):
            return ABSTAIN
    if isinstance(raw, (int, float)):
        try:
            v = int(raw)
        except (ValueError, OverflowError):
            return ABSTAIN
        if isinstance(raw, float) and v != raw:
            return ABSTAIN
        return v if v in ALLOWED else ABSTAIN
    if isinstance(raw, str):
        if raw in {'1', '2', '3', '4', '5'}:
            return int(raw)
        return ABSTAIN
    return ABSTAIN

def assert_valid(option: int, *, image_name: str='?') -> int:
    assert not isinstance(option, bool), f'validator bypass: bool {option!r} for {image_name!r}'
    assert type(option) is int, f'validator bypass: non-int {option!r} ({type(option).__name__}) for {image_name!r}'
    assert option in ALLOWED, f'validator bypass: {option!r} for {image_name!r}'
    return option
