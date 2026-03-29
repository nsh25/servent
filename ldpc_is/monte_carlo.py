from __future__ import annotations

import numpy as np

from .awgn import add_awgn, channel_llr_awgn
from .decoder_interface import LDPCDecoderInterface


def _ci_binomial(phat: float, n: int) -> tuple[float, float]:
    se = np.sqrt(max(phat * (1.0 - phat), 0.0) / n)
    return phat - 1.96 * se, phat + 1.96 * se


def estimate_error_rates_mc(
    x0: np.ndarray,
    sigma: float,
    decoder: LDPCDecoderInterface,
    num_frames: int,
    rng: np.random.Generator,
) -> dict:
    """Plain Monte Carlo BER/FER estimator under AWGN channel."""
    n = len(x0)
    fe = 0
    be = 0
    for _ in range(num_frames):
        y = add_awgn(x0, sigma, rng)
        dec = decoder.decode(channel_llr_awgn(y, sigma))
        fe += int(dec.frame_error)
        be += int(dec.num_bit_errors)

    fer = fe / num_frames
    ber = be / (n * num_frames)
    return {
        "FER_hat": fer,
        "BER_hat": ber,
        "num_frames": num_frames,
        "frame_errors": fe,
        "bit_errors": be,
        "FER_ci95": _ci_binomial(fer, num_frames),
        "BER_ci95": _ci_binomial(ber, n * num_frames),
    }
