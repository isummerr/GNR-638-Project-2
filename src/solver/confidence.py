from __future__ import annotations
import os
from .output_validator import ABSTAIN
DEFAULT_TAU = 0.1
ENV_TAU_KEY = 'CONFIDENCE_TAU'

def _resolve_tau(tau: float | None) -> float:
    if tau is not None:
        return float(tau)
    raw = os.environ.get(ENV_TAU_KEY)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_TAU

class ConfidenceGate:

    def __init__(self, tau: float | None=None) -> None:
        self.tau = _resolve_tau(tau)

    def gate(self, option_int: int, margin: float) -> int:
        if option_int == ABSTAIN:
            return ABSTAIN
        if option_int not in (1, 2, 3, 4):
            return ABSTAIN
        if margin >= self.tau:
            return option_int
        return ABSTAIN
