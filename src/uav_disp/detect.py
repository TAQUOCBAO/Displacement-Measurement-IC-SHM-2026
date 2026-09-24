"""Checkerboard target localization in a grayscale frame (no cv2).

Both targets are 2x2 checkerboards: two black squares on the diagonal of a
white board. Strategy: local threshold -> connected components -> pick the two
similar-area black squares nearest the approximate center. Their centroids give
the board center (midpoint), the checker square size in px (centroid distance /
sqrt(2)) and the board rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Board:
    center: np.ndarray        # (x, y) board center = X-junction of the 2x2 pattern
    square_px: float          # side of one checker square in px
    angle_deg: float          # rotation of the board relative to axis-aligned
    black_centroids: np.ndarray  # (2, 2) centroids of the two black squares
    white_centroids: np.ndarray  # (2, 2) mirror points on the white diagonal

    def query_points(self) -> np.ndarray:
        """9 well-textured points: center, square centroids, edge T-junctions."""
        diag = self.black_centroids[0] - self.center
        diag = diag / np.linalg.norm(diag)
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        u = np.array([c * diag[0] + s * diag[1], -s * diag[0] + c * diag[1]])
        v = np.array([c * diag[0] - s * diag[1], s * diag[0] + c * diag[1]])
        mids = self.center + self.square_px * np.array([u, -u, v, -v])
        return np.vstack([self.center, self.black_centroids, self.white_centroids, mids])


def find_board(gray: np.ndarray, approx_xy: tuple[float, float], approx_square_px: float) -> Board:
    x0, y0 = approx_xy
    r = int(approx_square_px * 3)
    ys, xs = slice(max(0, int(y0) - r), int(y0) + r), slice(max(0, int(x0) - r), int(x0) + r)
    roi = gray[ys, xs].astype(np.float32)

    thresh = roi.mean() * 0.6  # black squares sit far below the white board / red cable
    labels, n = ndimage.label(roi < thresh)
    if n < 2:
        raise ValueError("fewer than two dark components near target")

    areas = ndimage.sum_labels(np.ones_like(roi), labels, index=range(1, n + 1))
    centroids = np.array(ndimage.center_of_mass(np.ones_like(roi), labels, range(1, n + 1)))
    centroids = centroids[:, ::-1]  # (row, col) -> (x, y)

    a_exp = approx_square_px**2
    ok = (areas > 0.3 * a_exp) & (areas < 3.0 * a_exp)
    if ok.sum() < 2:
        raise ValueError("no square-sized dark components near target")
    cand = np.where(ok)[0]
    center_local = np.array([x0 - xs.start, y0 - ys.start])
    cand = cand[np.argsort(np.linalg.norm(centroids[cand] - center_local, axis=1))[:2]]

    c1, c2 = centroids[cand[0]], centroids[cand[1]]
    center = (c1 + c2) / 2
    d = np.linalg.norm(c2 - c1)
    square_px = d / np.sqrt(2)
    # black diagonal is at 45 deg on an axis-aligned board
    angle_deg = np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0])) - 45.0
    perp = np.array([-(c2 - center)[1], (c2 - center)[0]])
    w1, w2 = center + perp, center - perp

    off = np.array([xs.start, ys.start], dtype=float)
    return Board(
        center=center + off,
        square_px=float(square_px),
        angle_deg=float(angle_deg % 90.0),
        black_centroids=np.vstack([c1, c2]) + off,
        white_centroids=np.vstack([w1, w2]) + off,
    )
