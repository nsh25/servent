from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .awgn import channel_llr_awgn
from .decoder_interface import LDPCDecoderInterface
from .mixture_is import importance_weight, sample_from_mixture

logger = logging.getLogger(__name__)


def _mean_var_ci(x: np.ndarray) -> dict[str, float]:
    n = len(x)
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else float("nan")
    return {
        "mean": mean,
        "var": var,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def estimate_error_rates_is(
    x0: np.ndarray,
    sigma: float,
    decoder: LDPCDecoderInterface,
    centers: np.ndarray,
    mixture_probs: np.ndarray,
    num_samples: int,
    batch_size: int,
    rng: np.random.Generator,
    return_diagnostics: bool = False,
) -> dict:
    """Estimate FER/BER with mixture-IS and likelihood-ratio weighting."""
    n = len(x0)
    weighted_fe = np.empty(num_samples, dtype=np.float64)
    weighted_be = np.empty(num_samples, dtype=np.float64)
    weights = np.empty(num_samples, dtype=np.float64)
    comps = np.empty(num_samples, dtype=int)
    raw_fail = 0
    row_list = []

    cursor = 0
    while cursor < num_samples:
        bs = min(batch_size, num_samples - cursor)
        y, cidx = sample_from_mixture(centers, mixture_probs, sigma, bs, rng)
        w = importance_weight(y, x0, centers, mixture_probs, sigma)
        for j in range(bs):
            llr = channel_llr_awgn(y[j], sigma)
            dec = decoder.decode(llr)
            fe = 1.0 if dec.frame_error else 0.0
            be = float(dec.num_bit_errors)
            weighted_fe[cursor + j] = fe * w[j]
            weighted_be[cursor + j] = be * w[j]
            weights[cursor + j] = w[j]
            comps[cursor + j] = cidx[j]
            raw_fail += int(fe)
            if return_diagnostics:
                row_list.append({"idx": cursor + j, "comp": int(cidx[j]), "w": float(w[j]), "fe": fe, "be": be})
        cursor += bs

    fer_stats = _mean_var_ci(weighted_fe)
    ber_stats = _mean_var_ci(weighted_be / n)
    ess = float((weights.sum() ** 2) / np.sum(weights**2)) if np.sum(weights**2) > 0 else 0.0
    cov_w = float(np.std(weights) / np.mean(weights)) if np.mean(weights) > 0 else float("inf")
    contrib = pd.DataFrame({"comp": comps, "wfe": weighted_fe, "wbe": weighted_be}).groupby("comp", as_index=False).sum()

    out = {
        "FER_hat": fer_stats["mean"],
        "BER_hat": ber_stats["mean"],
        "FER_se": fer_stats["se"],
        "BER_se": ber_stats["se"],
        "FER_ci95": (fer_stats["ci95_low"], fer_stats["ci95_high"]),
        "BER_ci95": (ber_stats["ci95_low"], ber_stats["ci95_high"]),
        "effective_sample_size": ess,
        "weight_cov": cov_w,
        "raw_failure_count": raw_fail,
        "weighted_component_contrib": contrib,
        "weights_finite": bool(np.all(np.isfinite(weights))),
    }
    if return_diagnostics:
        out["sample_diagnostics"] = pd.DataFrame(row_list)
    return out
