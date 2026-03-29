from __future__ import annotations

import numpy as np


def sigma_from_ebn0_db(ebn0_db: float, rate: float = 1.0) -> float:
    """Return AWGN sigma for BPSK given Eb/N0 in dB and code rate."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return float(np.sqrt(1.0 / (2.0 * rate * ebn0)))


def add_awgn(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add iid Gaussian noise N(0, sigma^2) to x."""
    return np.asarray(x, dtype=np.float64) + rng.normal(0.0, sigma, size=np.shape(x))


def channel_llr_awgn(y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute channel LLRs for BPSK/AWGN: LLR = 2y/sigma^2."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    y = np.asarray(y, dtype=np.float64)
    return (2.0 / (sigma * sigma)) * y
