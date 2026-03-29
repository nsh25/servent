from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .types import TrappingSetRecord

_TRAP_RE = re.compile(r"^\s*\((\d+)\s*,\s*(\d+)\)\s+(.*)$")


def normalize_direction(v: np.ndarray) -> np.ndarray:
    """Normalize a direction to unit Euclidean norm."""
    v = np.asarray(v, dtype=np.float64)
    nv = np.linalg.norm(v)
    if nv <= 0:
        raise ValueError("direction has zero norm")
    return v / nv


def ts_indicator(ts: np.ndarray, n: int) -> np.ndarray:
    """Build a binary indicator vector for a trapping-set support."""
    ts = np.asarray(ts, dtype=int)
    out = np.zeros(n, dtype=np.float64)
    out[ts] = 1.0
    return out


def ts_direction(ts: np.ndarray, n: int, mode: str = "unsigned", sign_pattern: np.ndarray | None = None) -> np.ndarray:
    """Build a normalized direction from trapping-set support."""
    ind = ts_indicator(ts, n)
    if mode == "unsigned":
        return normalize_direction(ind)
    if mode == "signed":
        if sign_pattern is None:
            raise ValueError("signed mode requires sign_pattern")
        sp = np.asarray(sign_pattern, dtype=np.float64)
        if sp.shape != ind.shape:
            raise ValueError("sign_pattern shape mismatch")
        return normalize_direction(ind * sp)
    raise ValueError(f"unsupported mode={mode}")


def validate_trapping_sets(ts_list: list[np.ndarray], n: int) -> None:
    """Validate trapping-set indices are unique and inside [0, n-1]."""
    for i, ts in enumerate(ts_list):
        t = np.asarray(ts, dtype=int)
        if t.ndim != 1:
            raise ValueError(f"TS #{i} must be 1D")
        if len(np.unique(t)) != len(t):
            raise ValueError(f"TS #{i} has duplicate indices")
        if np.any(t < 0) or np.any(t >= n):
            raise ValueError(f"TS #{i} contains out-of-range nodes for n={n}")


def load_trap_file(path: str, n: int | None = None) -> list[TrappingSetRecord]:
    """Parse .trap text files with lines: '(a, b) v1 v2 ... va' (1-based indices)."""
    records: list[TrappingSetRecord] = []
    for line_no, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _TRAP_RE.match(line)
        if not m:
            raise ValueError(f"Malformed .trap line {line_no}: '{raw}'")
        a = int(m.group(1))
        b = int(m.group(2))
        tail = m.group(3).strip()
        if not tail:
            raise ValueError(f"Line {line_no}: missing node indices")
        tokens = tail.split()
        if len(tokens) != a:
            raise ValueError(f"Line {line_no}: expected {a} indices, got {len(tokens)}")
        nodes1 = np.array([int(t) for t in tokens], dtype=int)
        if np.any(nodes1 <= 0):
            raise ValueError(f"Line {line_no}: .trap indices must be 1-based positive integers")
        if n is not None and np.any(nodes1 > n):
            raise ValueError(f"Line {line_no}: node index exceeds n={n}")
        nodes0 = nodes1 - 1
        records.append(TrappingSetRecord(a=a, b=b, nodes_1based=nodes1, nodes_0based=nodes0))
    return records


def filter_trapping_sets(
    records: list[TrappingSetRecord],
    a_values: set[int] | None = None,
    b_values: set[int] | None = None,
    max_count: int | None = None,
    sort_ab: bool = True,
) -> list[TrappingSetRecord]:
    """Filter/sort trapping-set records by (a,b) labels and count."""
    out = [
        r
        for r in records
        if (a_values is None or r.a in a_values) and (b_values is None or r.b in b_values)
    ]
    if sort_ab:
        out = sorted(out, key=lambda r: (r.a, r.b))
    if max_count is not None:
        out = out[:max_count]
    return out


def extract_ts_node_lists(records: list[TrappingSetRecord]) -> list[np.ndarray]:
    """Extract 0-based node lists from trapping-set records."""
    return [r.nodes_0based.copy() for r in records]


def _load_json(path: Path) -> list[np.ndarray]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("JSON TS file must contain a list")
    return [np.asarray(x, dtype=int) for x in data]


def _load_csv(path: Path) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            out.append(np.asarray([int(x) for x in row], dtype=int))
    return out


def load_trapping_sets(path: str, n: int | None = None) -> list[np.ndarray]:
    """Load trapping sets from .trap/.json/.csv and return 0-based arrays."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".trap":
        return extract_ts_node_lists(load_trap_file(path, n=n))
    if suffix == ".json":
        return _load_json(p)
    if suffix == ".csv":
        return _load_csv(p)
    raise ValueError(f"Unsupported trapping-set file extension: {suffix}")
