from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CodeConfig:
    n: int
    k: int | None = None
    H_path: str | None = None


@dataclass(slots=True)
class DecoderConfig:
    max_iters: int = 50
    decoder_name: str = "dummy"
    llr_clip: float | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ISConfig:
    ebn0_db: float
    num_samples: int
    mixture_weights: str = "uniform"
    boundary_search_tol: float = 1e-3
    boundary_search_max_iter: int = 40
    initial_bracket_low: float = 0.0
    initial_bracket_high: float = 1.0
    center_scaling: float = 1.0
    use_ts_sign_pattern: bool = False
    rng_seed: int = 0
    batch_size: int = 256


@dataclass(slots=True)
class MCConfig:
    ebn0_db: float
    num_frames: int
    rng_seed: int = 0


@dataclass(slots=True)
class RunConfig:
    output_dir: str = "outputs"
    save_plots: bool = True
    save_csv: bool = True
    verbose: bool = True
