from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from .types import DecoderResult


class LDPCDecoderInterface(Protocol):
    """Protocol for LDPC decoders operating on channel LLR vectors."""

    def decode(self, llr: np.ndarray) -> DecoderResult:
        """Decode an LLR vector and return a standardized DecoderResult."""


@dataclass(slots=True)
class DummyThresholdDecoder:
    """Simple deterministic demo decoder with thresholded failure region.

    This is not a real LDPC decoder; it is a pluggable placeholder for demos/tests.
    """

    fail_threshold: float = -0.25

    def decode(self, llr: np.ndarray) -> DecoderResult:
        llr = np.asarray(llr, dtype=np.float64)
        hard = (llr < 0.0).astype(np.int8)
        # deterministic failure surrogate
        frame_error = bool(np.min(llr) < self.fail_threshold)
        if frame_error:
            hard = np.maximum(hard, (llr < self.fail_threshold).astype(np.int8))
        num_bit_errors = int(np.sum(hard))
        return DecoderResult(
            hard_bits=hard,
            num_bit_errors=num_bit_errors,
            frame_error=frame_error,
            converged=not frame_error,
            syndrome_weight=None,
            metadata={"decoder": "dummy_threshold"},
        )


@dataclass(slots=True)
class ExternalDecoderAdapter:
    """Adapter for a user-supplied decoder callable.

    The callable can return:
      - np.ndarray hard bits, or
      - dict with fields similar to DecoderResult, or
      - DecoderResult.
    """

    decoder_callable: Callable[[np.ndarray], Any]

    def decode(self, llr: np.ndarray) -> DecoderResult:
        out = self.decoder_callable(np.asarray(llr, dtype=np.float64))
        if isinstance(out, DecoderResult):
            return out
        if isinstance(out, np.ndarray):
            hard = out.astype(np.int8)
            nbe = int(np.sum(hard))
            return DecoderResult(hard_bits=hard, num_bit_errors=nbe, frame_error=nbe > 0, converged=True)
        if isinstance(out, dict):
            hard = np.asarray(out.get("hard_bits"), dtype=np.int8)
            nbe = int(out.get("num_bit_errors", np.sum(hard)))
            frame_error = bool(out.get("frame_error", nbe > 0))
            return DecoderResult(
                hard_bits=hard,
                num_bit_errors=nbe,
                frame_error=frame_error,
                converged=bool(out.get("converged", True)),
                syndrome_weight=out.get("syndrome_weight"),
                metadata=dict(out.get("metadata", {})),
            )
        raise TypeError("decoder_callable output type unsupported")
