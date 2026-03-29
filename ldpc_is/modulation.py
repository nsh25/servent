from __future__ import annotations

import numpy as np


def bpsk_map(bits: np.ndarray) -> np.ndarray:
    """Map binary bits to BPSK symbols using 0->+1, 1->-1."""
    bits = np.asarray(bits, dtype=np.int8)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bpsk_map expects binary bits in {0,1}")
    return 1.0 - 2.0 * bits.astype(np.float64)
