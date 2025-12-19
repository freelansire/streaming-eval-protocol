## Streaming Evaluation Protocol (Prequential + Latency + Memory + Calibration Drift)

A lightweight, reproducible **streaming evaluation protocol** for online/edge ML systems.  
It’s designed to *measure what actually matters in deployment*: **prequential performance**, **latency**, **memory footprint**, and **calibration stability under drift**.

---

### Why this exists
Batch evaluation can hide failure modes that appear in production: drifting data, unstable probabilities, slow per-event inference, and memory growth.  
This project provides a simple protocol you can reuse to evaluate online learners (e.g., SGD / online logistic models) in **test-then-train** settings.

---

### What’s included
### Prequential metrics (test → then train)
Measured at every timestep:
- **Log-loss** (streaming cross-entropy)
- **Brier score**
- **Accuracy**

#### Resource footprint
- **Latency per event (ms)** using `time.perf_counter()`
- **Memory footprint (MB)** via RSS (`psutil`)  
  - Fallback: `tracemalloc` peak memory estimate (if `psutil` isn’t available)

#### Calibration stability & drift diagnostics
Over a rolling window:
- **Rolling ECE** (Expected Calibration Error)
- **Calibration gap**: `| mean(p) - mean(y) |`
- **Calibration drift score**: `ECE_window - ECE_baseline`
- **Drift flag** when drift score crosses a threshold

---
### Quickstart

#### 1) Install
```bash
pip install -r requirements.txt

python src/run_demo.py
```
---
### Outputs (proof artifacts)

After running the demo, the project generates a complete set of **evidence artifacts** inside the `reports/` folder:
```bash
- `reports/streaming_metrics.csv` — per-timestep metrics *(prequential scoring + latency + memory + calibration)*
- `reports/summary.json` — aggregated summary statistics *(means, p95 latency, peak memory, drift flags, etc.)*
- `reports/prequential_logloss.png` — prequential log-loss over time *(test-then-train)*
- `reports/rolling_ece.png` — rolling calibration error (ECE) to monitor calibration stability and drift
- `reports/latency_ms.png` — per-event latency (milliseconds) across the stream
- `reports/memory_mb.png` — memory footprint (MB) across the stream
```
---
### How it works (conceptually)

This project follows a **prequential (test-then-train) streaming evaluation** protocol:
```bash
1. **Receive** the next event \(x_t\) from the stream  
2. **Predict** the probability \(p(y_t = 1 \mid x_t)\)  
3. **Score** the prediction against the ground truth label \(y_t\) (e.g., log-loss, Brier, accuracy)  
4. **Update** rolling calibration statistics (e.g., **ECE**, calibration gap)  
5. **Train incrementally** using `partial_fit` on \((x_t, y_t)\)  
6. **Record** deployment-relevant signals at step \(t\): **latency** and **memory footprint**
```
This mirrors real deployment conditions: the model is continuously adapting while being evaluated on the live stream.

---
#### Configuration knobs you can tune

You can adjust the main experimental and evaluation settings in:

- `src/run_demo.py`
- `src/streaming_eval.py`

#### Evaluation / calibration settings
- **`window`** — rolling window size used for calibration tracking (ECE + calibration gap).
- **`ece_bins`** — number of bins used to compute **ECE (Expected Calibration Error)**.
- **`warmup`** — minimum number of steps before establishing a baseline and enabling drift comparisons.
- **`drift_threshold`** — threshold on the calibration drift score that triggers a **drift flag**.

#### Stream generation settings
- **`drift_at`** — timestep at which concept drift is injected into the stream.
- **`noise`** — noise level in the synthetic sensor stream (higher = harder, noisier dynamics).

---
## Extending to your own model

You can plug in your own online/streaming model as long as it can be updated incrementally and can produce a prediction (preferably a probability).

### Requirements
1. **Incremental updates**
   - Your model should support `partial_fit(...)` (recommended), **or**
   - You should wrap it with a small adapter that performs incremental updates.

2. **Prediction interface**
   Implement **at least one** of the following methods:
   - `predict_proba(X)` *(preferred: returns calibrated probabilities)*
   - `decision_function(X)` *(accepted: converted to probability via sigmoid)*
   - `predict(X)` *(fallback: converted to a coarse probability estimate)*

### How probability is derived
The evaluator automatically selects the best available method to compute the class-1 probability:

1. If `predict_proba` exists → uses `proba[:, 1]` as `p(y=1)`
2. Else if `decision_function` exists → applies a sigmoid to map scores to `p(y=1)`
3. Else if `predict` exists → maps the hard label to a simple probability proxy

### Example (minimal adapter)
If your model does not support `partial_fit`, create a lightweight adapter:

```python
class OnlineAdapter:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def partial_fit(self, X, y, classes=None):
        # Replace with your incremental update logic (e.g., buffer + periodic retrain)
        self.model.fit(X, y)
        return self
```
---
#### Citation

```bibtex
@software{Streaming_Evaluation_Protocol,
  author  = {Moses, Samuel},
  title   = {Streaming Evaluation Protocol: Prequential Metrics, Latency, Memory Footprint, and Calibration Drift},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/freelansire/streaming-eval-protocol},
  note    = {GitHub repository}
}
```

---
### Repo layout
```text
streaming-eval-protocol/
├─ src/
│  ├─ run_demo.py           # runs end-to-end experiment + saves reports
│  ├─ streaming_eval.py     # core evaluator (prequential + resources + calibration drift)
│  ├─ data_stream.py        # drifting synthetic stream generator
│  ├─ metrics.py            # logloss, brier, ECE, calibration gap
│  └─ utils.py              # small helpers
├─ reports/                 # auto-generated outputs (CSV/JSON/plots)
├─ requirements.txt
└─ README.md
```
