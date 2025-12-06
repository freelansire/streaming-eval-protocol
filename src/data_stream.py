from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class StreamConfig:
    n_steps: int = 5000
    n_features: int = 8
    drift_at: int = 2500
    seed: int = 42
    noise: float = 1.0

def make_drifting_binary_stream(cfg: StreamConfig):
    """
    Binary classification stream with concept drift:
    - Before drift_at: class means centered around +m/-m
    - After drift_at: means shift + rotate slightly, making the boundary move.
    Yields (x, y, t, drift_flag).
    """
    rng = np.random.default_rng(cfg.seed)

    m1 = np.ones(cfg.n_features) * 0.8
    m0 = -np.ones(cfg.n_features) * 0.8

    # After drift: shift means + reweight a few dimensions
    m1_d = m1.copy(); m0_d = m0.copy()
    m1_d[: cfg.n_features // 3] += 1.2
    m0_d[: cfg.n_features // 3] -= 1.2

    # Slight rotation via random orthonormal-ish matrix (fast approximate)
    R = rng.normal(size=(cfg.n_features, cfg.n_features))
    # QR gives orthonormal Q
    Q, _ = np.linalg.qr(R)

    for t in range(cfg.n_steps):
        drift = t >= cfg.drift_at
        # Balanced labels
        y = int(rng.random() > 0.5)

        mean = (m1_d if drift else m1) if y == 1 else (m0_d if drift else m0)
        x = rng.normal(loc=mean, scale=cfg.noise, size=cfg.n_features).astype(np.float32)

        if drift:
            x = (Q @ x).astype(np.float32)

        yield x, y, t, drift
