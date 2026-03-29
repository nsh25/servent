from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

ArrayLike: TypeAlias = np.ndarray | list[float] | tuple[float, ...]
TrappingSet: TypeAlias = list[int] | np.ndarray


@dataclass(slots=True)
class DecoderResult:
    """Standardized decoder output for the IS/MC pipeline."""

    hard_bits: np.ndarray
    num_bit_errors: int
    frame_error: bool
    converged: bool
    syndrome_weight: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrappingSetRecord:
    """Parsed trapping-set entry from a .trap file."""

    a: int
    b: int
    nodes_1based: np.ndarray
    nodes_0based: np.ndarray
