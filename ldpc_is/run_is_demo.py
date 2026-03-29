from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .awgn import sigma_from_ebn0_db
from .boundary_search import rank_trapping_sets_by_boundary
from .decoder_interface import DummyThresholdDecoder
from .estimators import estimate_error_rates_is
from .mixture_is import build_mixture_centers, choose_mixture_weights
from .monte_carlo import estimate_error_rates_mc
from .trapping_sets import (
    extract_ts_node_lists,
    filter_trapping_sets,
    load_trap_file,
    ts_direction,
    validate_trapping_sets,
)
from .utils import all_zero_codeword, ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDPC trapping-set IS demo")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--H-path", type=str, default=None)
    p.add_argument("--ts-path", type=str, required=True)
    p.add_argument("--ebn0-db", type=float, required=True)
    p.add_argument("--num-is-samples", type=int, default=5000)
    p.add_argument("--num-mc-frames", type=int, default=2000)
    p.add_argument("--max-iters", type=int, default=50)
    p.add_argument("--top-m", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(True)
    out_dir = ensure_dir(args.output_dir)

    _, x0 = all_zero_codeword(args.n)
    sigma = sigma_from_ebn0_db(args.ebn0_db, rate=(args.k / args.n) if args.k else 1.0)

    recs = load_trap_file(args.ts_path, n=args.n)
    recs = filter_trapping_sets(recs, a_values=set(range(1, 11)), max_count=100)
    ts_list = extract_ts_node_lists(recs)
    validate_trapping_sets(ts_list, args.n)

    directions = [ts_direction(ts, args.n, mode="unsigned") for ts in ts_list]
    decoder = DummyThresholdDecoder()

    rank_df = rank_trapping_sets_by_boundary(
        ts_list=ts_list,
        directions=directions,
        x0=x0,
        sigma=sigma,
        decoder=decoder,
        low=0.0,
        high=1.0,
        tol=1e-3,
        max_iter=40,
        top_m=args.top_m,
    )
    active = rank_df[rank_df["active"]]
    if active.empty:
        raise RuntimeError("No active trapping sets found at this SNR")

    idx = active["ts_index"].to_numpy(dtype=int)
    alphas = active["alpha_star"].to_numpy(dtype=float)
    dirs = pd.Series(directions).iloc[idx].to_list()

    import numpy as np

    centers = build_mixture_centers(x0, np.vstack(dirs), alphas, center_scaling=1.0)
    probs = choose_mixture_weights(alphas, mode="distance_based", beta=1.0)

    rng = np.random.default_rng(args.seed)
    is_res = estimate_error_rates_is(x0, sigma, decoder, centers, probs, args.num_is_samples, batch_size=256, rng=rng, return_diagnostics=True)
    mc_res = estimate_error_rates_mc(x0, sigma, decoder, args.num_mc_frames, np.random.default_rng(args.seed + 1))

    rank_df.to_csv(out_dir / "boundary_ranking.csv", index=False)
    pd.DataFrame([is_res | {"ebn0_db": args.ebn0_db}]).drop(columns=["weighted_component_contrib", "sample_diagnostics"]).to_csv(out_dir / "is_summary.csv", index=False)
    pd.DataFrame([mc_res | {"ebn0_db": args.ebn0_db}]).to_csv(out_dir / "mc_summary.csv", index=False)
    is_res["weighted_component_contrib"].to_csv(out_dir / "is_component_contrib.csv", index=False)
    is_res["sample_diagnostics"].to_csv(out_dir / "is_sample_diagnostics.csv", index=False)

    print("=== LDPC IS Demo Summary ===")
    print(f"Eb/N0={args.ebn0_db:.2f} dB sigma={sigma:.4f}")
    print(f"IS FER={is_res['FER_hat']:.4e} BER={is_res['BER_hat']:.4e} ESS={is_res['effective_sample_size']:.1f}")
    print(f"MC FER={mc_res['FER_hat']:.4e} BER={mc_res['BER_hat']:.4e}")
    print(f"Saved outputs to {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
