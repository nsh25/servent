from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


def setup_logging(verbose: bool = True) -> None:
    """Configure package-level logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def ensure_dir(path: str) -> Path:
    """Create a directory tree if missing and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def all_zero_codeword(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return all-zero bits and corresponding all-ones BPSK symbols."""
    bits = np.zeros(n, dtype=np.int8)
    x0 = np.ones(n, dtype=np.float64)
    return bits, x0


def suggest_ts_expansion(ess: float, num_samples: int, max_component_share: float) -> str:
    """Generate a simple diagnostic suggestion for IS health."""
    if ess < 0.05 * num_samples:
        return "ESS is very low; add more dominant trapping sets or tune center_scaling/weights."
    if max_component_share > 0.8:
        return "Single component dominates weighted mass; consider adding TS diversity."
    return "IS diagnostics look healthy."
