from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .awgn import channel_llr_awgn
from .decoder_interface import LDPCDecoderInterface

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BoundarySearchResult:
    alpha_star: float
    boundary_point_y: np.ndarray
    boundary_point_llr: np.ndarray
    found_failure: bool
    num_decoder_calls: int
    bracket_history: list[tuple[float, float, bool, bool]]


def _fails(y: np.ndarray, sigma: float, decoder: LDPCDecoderInterface) -> tuple[bool, int | None]:
    res = decoder.decode(channel_llr_awgn(y, sigma))
    fail = bool(res.frame_error or (not res.converged))
    return fail, res.syndrome_weight


def find_failure_boundary(
    direction: np.ndarray,
    x0: np.ndarray,
    sigma: float,
    decoder: LDPCDecoderInterface,
    low: float,
    high: float,
    tol: float,
    max_iter: int,
    max_expand: int = 25,
) -> dict:
    """Binary-search smallest alpha where decoding y=x0-alpha*direction fails."""
    d = np.asarray(direction, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    history: list[tuple[float, float, bool, bool]] = []
    calls = 0

    low_fail, _ = _fails(x0 - low * d, sigma, decoder)
    calls += 1
    if low_fail:
        logger.warning("Failure already at low bound; setting alpha*=low")
        yb = x0 - low * d
        return BoundarySearchResult(low, yb, channel_llr_awgn(yb, sigma), True, calls, history).__dict__

    hi = high
    hi_fail = False
    for _ in range(max_expand):
        hi_fail, _ = _fails(x0 - hi * d, sigma, decoder)
        calls += 1
        if hi_fail:
            break
        hi *= 2.0
    if not hi_fail:
        yb = x0 - hi * d
        return BoundarySearchResult(float("nan"), yb, channel_llr_awgn(yb, sigma), False, calls, history).__dict__

    lo = low
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        lo_fail, _ = _fails(x0 - lo * d, sigma, decoder)
        mid_fail, _ = _fails(x0 - mid * d, sigma, decoder)
        calls += 2
        history.append((lo, hi, lo_fail, mid_fail))
        if mid_fail:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= tol:
            break

    alpha_star = hi
    yb = x0 - alpha_star * d
    return BoundarySearchResult(alpha_star, yb, channel_llr_awgn(yb, sigma), True, calls, history).__dict__


def rank_trapping_sets_by_boundary(
    ts_list: list[np.ndarray],
    directions: list[np.ndarray],
    x0: np.ndarray,
    sigma: float,
    decoder: LDPCDecoderInterface,
    low: float,
    high: float,
    tol: float,
    max_iter: int,
    top_m: int | None = None,
) -> pd.DataFrame:
    """Compute and rank trapping sets by boundary squared distance alpha*^2."""
    rows = []
    for i, (ts, d) in enumerate(zip(ts_list, directions, strict=True)):
        res = find_failure_boundary(d, x0, sigma, decoder, low, high, tol, max_iter)
        alpha = float(res["alpha_star"])
        active = bool(res["found_failure"])
        rows.append(
            {
                "ts_index": i,
                "a": int(len(ts)),
                "alpha_star": alpha,
                "squared_distance": alpha * alpha if np.isfinite(alpha) else np.inf,
                "active": active,
                "decoder_calls": int(res["num_decoder_calls"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["active", "squared_distance", "a"], ascending=[False, True, True])
    if top_m is not None:
        df = df.head(top_m)
    return df.reset_index(drop=True)
