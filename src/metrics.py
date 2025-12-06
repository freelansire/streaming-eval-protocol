from __future__ import annotations
import numpy as np

def safe_logloss(y_true: int, p1: float, eps: float = 1e-12) -> float:
    p1 = float(np.clip(p1, eps, 1.0 - eps))
    return -(y_true * np.log(p1) + (1 - y_true) * np.log(1 - p1))

def brier_score(y_true: int, p1: float) -> float:
    return float((p1 - y_true) ** 2)

def ece_binary(y_true: np.ndarray, p1: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for binary classification over a window.
    """
    y_true = y_true.astype(float)
    p1 = np.clip(p1.astype(float), 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p1 >= lo) & (p1 < hi) if i < n_bins - 1 else (p1 >= lo) & (p1 <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p1[mask]))
        acc = float(np.mean(y_true[mask]))
        w = float(np.sum(mask)) / n
        ece += w * abs(acc - conf)

    return float(ece)

def calibration_gap(y_true: np.ndarray, p1: np.ndarray) -> float:
    """
    Simple calibration stability proxy: |mean(p) - mean(y)|
    """
    if len(y_true) == 0:
        return 0.0
    return float(abs(np.mean(p1) - np.mean(y_true)))
