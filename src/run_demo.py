from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from data_stream import StreamConfig, make_drifting_binary_stream
from streaming_eval import StreamingEvaluator, EvalConfig
from utils import ensure_dir, dump_json

def main():
    # --- Config ---
    stream_cfg = StreamConfig(n_steps=5000, n_features=8, drift_at=2500, seed=42, noise=1.0)
    eval_cfg = EvalConfig(window=300, ece_bins=10, drift_threshold=0.04, warmup=80)

    # --- Model: online linear classifier ---
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=42,
    )
    classes = np.array([0, 1], dtype=int)

    evaluator = StreamingEvaluator(eval_cfg)

    rows = []
    # Important: SGD needs initial partial_fit call with classes
    x0, y0, t0, drift0 = next(make_drifting_binary_stream(stream_cfg))
    model.partial_fit(x0.reshape(1, -1), np.asarray([y0]), classes=classes)

    # Evaluate remaining stream prequentially
    for x, y, t, drift in make_drifting_binary_stream(stream_cfg):
        out = evaluator.step(model, x, y, classes=classes)
        out["t"] = t
        out["true_drift_region"] = bool(drift)
        rows.append(out)

    df = pd.DataFrame(rows)
    reports = ensure_dir("./reports")
    csv_path = reports / "streaming_metrics.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "n_steps": len(df),
        "mean_logloss": float(df["logloss"].mean()),
        "mean_brier": float(df["brier"].mean()),
        "mean_acc": float(df["acc"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "peak_mem_mb": float(df["mem_mb"].max()),
        "drift_flags_total": int(df["cal_drift_flag"].sum()),
        "drift_threshold": eval_cfg.drift_threshold,
        "window": eval_cfg.window,
    }
    dump_json(summary, reports / "summary.json")

    # --- Plots ---
    # 1) Prequential loss + drift score
    plt.figure(figsize=(11, 4))
    plt.plot(df["t"], df["logloss"], alpha=0.7)
    plt.axvline(stream_cfg.drift_at, linestyle="--")
    plt.title("Prequential Log-Loss (test-then-train)")
    plt.xlabel("t")
    plt.ylabel("logloss")
    plt.tight_layout()
    plt.savefig(reports / "prequential_logloss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(df["t"], df["ece_win"], alpha=0.8, label="ECE (rolling)")
    plt.axvline(stream_cfg.drift_at, linestyle="--")
    plt.title("Calibration Stability: Rolling ECE")
    plt.xlabel("t")
    plt.ylabel("ECE")
    plt.tight_layout()
    plt.savefig(reports / "rolling_ece.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(df["t"], df["latency_ms"], alpha=0.6)
    plt.axvline(stream_cfg.drift_at, linestyle="--")
    plt.title("Latency per Event (ms)")
    plt.xlabel("t")
    plt.ylabel("latency_ms")
    plt.tight_layout()
    plt.savefig(reports / "latency_ms.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(df["t"], df["mem_mb"], alpha=0.6)
    plt.axvline(stream_cfg.drift_at, linestyle="--")
    plt.title("Memory Footprint (MB)")
    plt.xlabel("t")
    plt.ylabel("mem_mb")
    plt.tight_layout()
    plt.savefig(reports / "memory_mb.png", dpi=160)
    plt.close()

    print("✅ Done.")
    print(f"- CSV: {csv_path}")
    print(f"- Summary: {reports / 'summary.json'}")
    print(f"- Plots saved in: {reports.resolve()}")

if __name__ == "__main__":
    main()
