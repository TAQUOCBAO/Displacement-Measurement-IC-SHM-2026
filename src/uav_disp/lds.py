"""LDS reference data: xlsx -> npy cache, loading, decimation.

The xlsx holds a single column A: header "LDS" in A1, then ~600k displacement
samples recorded at 10 kHz (no time column). Values are assumed to be mm.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
from scipy import signal

LDS_FS = 10_000.0  # Hz


def convert_xlsx_to_npy(xlsx_path: str | Path, npy_path: str | Path) -> np.ndarray:
    """Parse column A of sheet1 directly from the xlsx XML (pandas is ~100x slower)."""
    with zipfile.ZipFile(xlsx_path) as z:
        xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8", errors="ignore")

    cells = re.findall(r'<c r="A(\d+)"([^>]*)>(?:<[^v][^>]*/?>)*<v>([^<]+)</v>', xml)
    rows_vals = [(int(r), float(v)) for r, attrs, v in cells if 't="s"' not in attrs]
    if not rows_vals:
        raise ValueError(f"no numeric cells found in column A of {xlsx_path}")

    rows = np.array([r for r, _ in rows_vals])
    vals = np.array([v for _, v in rows_vals])
    order = np.argsort(rows)
    rows, vals = rows[order], vals[order]
    if not np.array_equal(np.diff(rows), np.ones(len(rows) - 1, dtype=rows.dtype)):
        raise ValueError("column A has gaps; parsing missed rows")

    np.save(npy_path, vals)
    return vals


def load(npy_path: str | Path, xlsx_path: str | Path | None = None) -> np.ndarray:
    npy_path = Path(npy_path)
    if npy_path.exists():
        return np.load(npy_path)
    if xlsx_path is None:
        raise FileNotFoundError(npy_path)
    return convert_xlsx_to_npy(xlsx_path, npy_path)


def decimate_to(x: np.ndarray, fs_out: float, fs_in: float = LDS_FS) -> np.ndarray:
    """Anti-aliased decimation, e.g. 10 kHz -> 50 Hz (factor 200 = 8 * 25)."""
    factor = fs_in / fs_out
    if abs(factor - round(factor)) > 1e-9:
        raise ValueError(f"non-integer decimation factor {factor}")
    factor = int(round(factor))
    # scipy recommends cascading for factors > 13
    for f in _factorize(factor):
        x = signal.decimate(x, f, ftype="fir", zero_phase=True)
    return x


def _factorize(n: int, cap: int = 10) -> list[int]:
    out = []
    while n > 1:
        for f in range(min(cap, n), 1, -1):
            if n % f == 0:
                out.append(f)
                n //= f
                break
        else:
            out.append(n)
            n = 1
    return out
