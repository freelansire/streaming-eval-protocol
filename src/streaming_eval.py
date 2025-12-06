from __future__ import annotations
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from metrics import safe_logloss, brier_score, ece_binary, calibration_gap

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class EvalConfig:
    window: int = 300                 # rolling window for calibration drift
    ece_bins: int = 10
    drift_threshold: float = 0.05     # drift flag if ECE increases by this much vs baseline
    warmup: int = 50                  # steps before starting drift comparisons


class StreamingEvaluator:
    """
    Prequential evaluator:
    - For each event: predict -> score -> update rolling stats -> train (partial_fit)
    Tracks: logloss, brier, accuracy, latency, memory, calibration drift.
    """
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self._y_win: List[int] = []
        self._p_win: List[float] = []
        self._ece_baseline: Optional[float] = None

        tracemalloc.start()

    def _rss_mb(self) -> float:
        if psutil is not None:
            proc = psutil.Process()
            return float(proc.memory_info().rss) / (1024.0 ** 2)
        # fallback: tracemalloc peak → approximate
        current, peak = tracemalloc.get_traced_memory()
        return float(peak) / (1024.0 ** 2)

    def step(
        self,
        model: Any,
        x: np.ndarray,
        y: int,
        classes: np.ndarray,
    ) -> Dict[str, Any]:
        """
        One streaming step (prequential):
        1) predict
        2) score
        3) update calibration window
        4) partial_fit
        """
        start = time.perf_counter()

        # --- Predict probability of class 1 (best-effort) ---
        p1 = self._predict_proba_1(model, x)

        # --- Scoring ---
        yhat = int(p1 >= 0.5)
        ll = safe_logloss(y, p1)
        bs = brier_score(y, p1)
        acc = 1.0 if yhat == y else 0.0

        # --- Update rolling window ---
        self._y_win.append(y)
        self._p_win.append(p1)
        if len(self._y_win) > self.cfg.window:
            self._y_win.pop(0)
            self._p_win.pop(0)

        y_arr = np.asarray(self._y_win, dtype=int)
        p_arr = np.asarray(self._p_win, dtype=float)

        ece = ece_binary(y_arr, p_arr, n_bins=self.cfg.ece_bins)
        gap = calibration_gap(y_arr, p_arr)

        # establish baseline after warmup with sufficient window
        drift_score = 0.0
        drift_flag = False
        if len(y_arr) >= max(self.cfg.warmup, self.cfg.window // 2):
            if self._ece_baseline is None:
                self._ece_baseline = ece
            drift_score = float(ece - self._ece_baseline)
            drift_flag = drift_score >= self.cfg.drift_threshold

        # --- Train (online update) ---
        # SGDClassifier requires 2D input
        model.partial_fit(x.reshape(1, -1), np.asarray([y]), classes=classes)

        latency_ms = (time.perf_counter() - start) * 1000.0
        mem_mb = self._rss_mb()

        return {
            "y": y,
            "p1": p1,
            "yhat": yhat,
            "logloss": ll,
            "brier": bs,
            "acc": acc,
            "latency_ms": latency_ms,
            "mem_mb": mem_mb,
            "ece_win": ece,
            "cal_gap_win": gap,
            "cal_drift_score": drift_score,
            "cal_drift_flag": drift_flag,
        }

    @staticmethod
    def _predict_proba_1(model: Any, x: np.ndarray) -> float:
        # If model supports predict_proba:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x.reshape(1, -1))[0]
            # assume class ordering corresponds to model.classes_
            if len(proba) == 2:
                return float(proba[1])
        # If model supports decision_function:
        if hasattr(model, "decision_function"):
            s = float(model.decision_function(x.reshape(1, -1))[0])
            # sigmoid to map scores -> probability
            return float(1.0 / (1.0 + np.exp(-s)))
        # fallback: predict
        if hasattr(model, "predict"):
            yhat = int(model.predict(x.reshape(1, -1))[0])
            return 0.9 if yhat == 1 else 0.1
        return 0.5
