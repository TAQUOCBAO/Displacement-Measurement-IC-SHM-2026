"""Scene-adaptive sub-pixel refinement network (S4).

A small CNN regresses the residual sub-pixel offset between a frame-0 template
patch and the current patch cut at the foundation tracker's coarse position:
the foundation model supplies robust association, the network supplies
metrological precision. Training is self-supervised on the scene itself —
Fourier-shifted real patches give unlimited exact labels — so no external data
or annotation is needed and the network adapts to this scene's texture, blur,
and noise statistics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .synth import fourier_shift

PATCH = 48       # patch side seen by the network
SRC = PATCH + 16  # padded source patch; shifts stay well inside the margin
MAX_SHIFT = 2.0


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):  # x: (B, 2, PATCH, PATCH), normalized
        return self.head(self.net(x))


def normalize(patch: np.ndarray | torch.Tensor):
    p = patch if isinstance(patch, torch.Tensor) else torch.from_numpy(patch.astype(np.float32))
    p = p.float()
    mu = p.mean(dim=(-2, -1), keepdim=True)
    sd = p.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    return (p - mu) / sd


def _center_crop(img: np.ndarray) -> np.ndarray:
    m = (SRC - PATCH) // 2
    return img[m : m + PATCH, m : m + PATCH]


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG round-trip: stationary block-structured codec artifacts that do NOT
    translate with the content — the dominant real-video degradation regime."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(buf)).astype(np.float64)


def make_batch(sources: list[np.ndarray], batch: int, rng: np.random.Generator):
    """Training pairs from padded source patches (SRC x SRC, float grayscale)."""
    xs = np.empty((batch, 2, PATCH, PATCH), np.float32)
    ys = np.empty((batch, 2), np.float32)
    for i in range(batch):
        src = sources[rng.integers(len(sources))]
        r0 = rng.uniform(-0.5, 0.5, 2)
        delta = rng.uniform(-MAX_SHIFT, MAX_SHIFT, 2)
        t = _center_crop(fourier_shift(src, *r0))
        p = _center_crop(fourier_shift(src, *(r0 + delta)))
        sigma = rng.uniform(0.5, 3.0)
        gain = rng.uniform(0.9, 1.1)
        bias = rng.uniform(-8, 8)
        t = t * gain + bias + rng.normal(0, sigma, t.shape)
        p = p * gain + bias + rng.normal(0, sigma, p.shape)
        if rng.random() < 0.7:  # independent codec realizations per patch
            q = int(rng.integers(60, 92))
            t, p = _jpeg(t, q), _jpeg(p, q)
        xs[i, 0], xs[i, 1] = t, p
        ys[i] = delta
    return torch.from_numpy(xs), torch.from_numpy(ys)


def train(sources: list[np.ndarray], device: str, steps: int = 3000,
          batch: int = 64, seed: int = 0) -> tuple[RefineNet, float]:
    rng = np.random.default_rng(seed)
    model = RefineNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for step in range(steps):
        x, y = make_batch(sources, batch, rng)
        x, y = normalize(x).to(device), y.to(device)
        loss = (model(x) - y).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            print(f"  step {step}: L1 {loss.item():.4f} px")
    # held-out validation
    model.eval()
    rng_val = np.random.default_rng(seed + 1)
    with torch.no_grad():
        x, y = make_batch(sources, 512, rng_val)
        err = (model(normalize(x).to(device)) - y.to(device)).cpu().numpy()
    val_rmse = float(np.sqrt((err ** 2).mean()))
    print(f"  val RMSE {val_rmse:.4f} px/axis")
    return model, val_rmse


@torch.no_grad()
def refine_frame(model: RefineNet, device: str, gray: np.ndarray,
                 templates: torch.Tensor, coarse_xy: np.ndarray,
                 q_frac: np.ndarray) -> np.ndarray:
    """Refine all points of one frame.

    templates: (N, PATCH, PATCH) normalized frame-0 patches cut at round(q).
    coarse_xy: (N, 2) tracker positions this frame. q_frac: (N, 2) q - round(q).
    Returns (N, 2) refined positions: round(coarse) + q_frac + net delta.
    """
    n = len(coarse_xy)
    ip = np.rint(coarse_xy).astype(int)
    h = PATCH // 2
    cur = np.stack([gray[y - h : y + h, x - h : x + h] for x, y in ip]).astype(np.float32)
    x = torch.stack([templates, normalize(torch.from_numpy(cur))], dim=1).to(device)
    delta = model(x).cpu().numpy()
    return ip + q_frac + delta
