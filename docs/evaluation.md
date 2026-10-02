# Evaluation

The evaluation stage runs the exported ONNX model against held-out validation data and produces a DET (Detection Error Tradeoff) curve, AUT score, and summary metrics.

**Source:** `src/livekit/wakeword/eval/evaluate.py`
**CLI:** `livekit-wakeword eval <config>` or as step 6/6 of `livekit-wakeword run`

## Overview

```
ONNX model (.onnx) + validation features (.npy)
    │
    ▼
Run inference on positive & negative samples
    │
    ▼
Compute DET curve (FPR vs FNR across thresholds)
    │
    ▼
Compute AUT (Area Under the DET curve)
    │
    ▼
Find optimal threshold (maximize recall at target FPPH)
    │
    ▼
Save DET plot (.png) + metrics (.json)
```

## CLI Usage

**Evaluate after a full pipeline run:**

```bash
uv run livekit-wakeword eval configs/hey_livekit.yaml
```

**Evaluate a specific ONNX model** (e.g., from a different training run or an openWakeWord model):

```bash
uv run livekit-wakeword eval configs/hey_livekit.yaml -m /path/to/model.onnx
```

If `--model` is not specified, the default path `output/<model_name>/<model_name>.onnx` is used.

## Python API

```python
from pathlib import Path
from livekit.wakeword import load_config
from livekit.wakeword.eval.evaluate import run_eval

config = load_config("configs/hey_livekit.yaml")
model_path = Path("output/hey_livekit/hey_livekit.onnx")

results = run_eval(config, model_path)
# results = {
#     "aut": 0.0012,
#     "fpph": 0.08,
#     "recall": 0.861,
#     "accuracy": 0.93,
#     "threshold": 0.68,
#     "n_positive": 15000,
#     "n_negative": 45084,
#     "validation_hours": 25.05,
# }
```

## Metrics

### AUT — Area Under the DET Curve

The primary aggregate metric. Computed by integrating FNR as a function of FPR using the trapezoidal rule. Lower is better (0 = perfect separation).

AUT captures the full tradeoff between false positives and false negatives across all thresholds, making it useful for comparing models without committing to a specific operating point.

### DET Curve

The Detection Error Tradeoff curve plots False Positive Rate (x-axis) against False Negative Rate (y-axis) across 1001 thresholds from 0.0 to 1.0. A perfect model hugs the origin; a random classifier falls on the diagonal.

The DET curve is saved as `output/<model_name>/<model_name>_det.png` with an annotation box showing AUT, FPPH, recall, and the optimal threshold.

### FPPH — False Positives Per Hour

The number of false triggers per hour of negative audio. Computed as:

```
FPPH = (clip_false_accepts + stream_false_accepts) / (clip_hours + stream_hours)
```

- `clip_false_accepts = count(negative_clip_scores >= threshold)`, one decision per test clip,
  and `clip_hours = n_negative_clips × clip_duration / 3600`.
- The ~11 h validation stream is scored the way a live detector sees it: one 16-step window
  per 80 ms hop (stride 1), so every hop can fire. Consecutive hops above the threshold count as
  one false accept, and hops within `streaming_eval.debounce_seconds` (default 2 s) of a false
  accept are ignored, as in `WakeWordListener`. `stream_hours = n_hops × 0.08 / 3600`, and the
  stream-only rate is reported as `stream_fpph`.

Training validation (`_validate`, `_find_optimal_threshold`) uses the same computation.

### Recall

True positive rate at the optimal threshold:

```
Recall = mean(positive_scores >= threshold)
```

### Threshold Optimization

The evaluation uses `find_best_threshold()` to scan thresholds from 0.01 to 0.99 and select the one that **maximizes recall** while keeping **FPPH ≤ `target_fp_per_hour`** (from config). If no threshold meets the FPPH target, it falls back to maximizing balanced accuracy.

## Validation Data

### Sources

| Source | Path | Type |
|--------|------|------|
| Positive test clips | `output/<model>/positive_features_test.npy` | Wake word samples (generated + augmented) |
| Negative test clips | `output/<model>/negative_features_test.npy` | Adversarial negatives (phonetically similar) |
| General negatives | `data/features/validation_set_features.npy` | ~11 hrs of ACAV100M speech (downloaded via `setup`) |

All feature arrays have shape `(N, 16, 96)` — N clips × 16 embedding timesteps × 96-dim speech embeddings.

The general negative validation set (`validation_set_features.npy`) is stored as 2D `(N, 96)`, one embedding per 80 ms hop, and is scored over stride-1 windows (see FPPH above).

### Requirements

Evaluation requires that the data generation, augmentation, and feature extraction stages have been run first (to produce the `*_features_test.npy` files). The ACAV100M validation features are optional but recommended — without them, FPPH estimates are based only on adversarial negatives.

## Output Files

| File | Description |
|------|-------------|
| `<model_name>_det.png` | DET curve plot with metrics annotation |
| `<model_name>_eval.json` | Full metrics as JSON |

Example `_eval.json`:

```json
{
  "aut": 0.0012,
  "fpph": 0.08,
  "recall": 0.861,
  "accuracy": 0.93,
  "threshold": 0.68,
  "n_positive": 15000,
  "n_negative": 45084,
  "validation_hours": 25.05
}
```

## Comparing Models

The eval command accepts any ONNX model that takes `(1, 16, 96)` input and produces a `(1, 1)` score, making it useful for comparing models trained with different configurations or frameworks:

```bash
# Evaluate a livekit-wakeword model
uv run livekit-wakeword eval configs/hey_livekit.yaml -m models/conv_attention_medium.onnx

# Evaluate an openWakeWord model against the same validation set
uv run livekit-wakeword eval configs/hey_livekit.yaml -m models/hey_livekit_oww.onnx
```

This works because both livekit-wakeword and openWakeWord share the same frozen embedding front-end, producing identical `(16, 96)` feature matrices.

## Streaming Evaluation

**Source:** `src/livekit/wakeword/eval/streaming.py`
**CLI:** `livekit-wakeword eval <config> --streaming [--scorer streaming|window] [-t THRESHOLD]`

The clip evaluation above scores isolated 2-second windows. The streaming evaluation instead
lays the test clips end to end into audio streams and feeds them to the detector in 80 ms
frames, exactly like a microphone. Detections follow the listener's rules (threshold,
`min_consecutive`, debounce), so the numbers and the selected threshold apply to the deployed
path.

- `--scorer streaming` (default) uses `StreamingWakeWordModel` (cached embeddings).
- `--scorer window` re-runs `WakeWordModel.predict()` on the last 2 s at every hop (slower).

### Sets

Each set is one stream and is reported separately. By default:

| Set | Built from | Reported |
|-----|------------|----------|
| `positive_silence` | `positive_test` clips, silence between them | miss rate |
| `positive_speech` | `positive_test` clips, each preceded by a speech clip ending 0-400 ms before it | miss rate, false accepts on the preceding speech |
| `near_miss` | `negative_test` clips (adversarial phrases) | share of clips that fired, false accepts per hour |
| `validation_stream` | `validation_set_features.npy`, every hop | false accepts per hour |

A positive clip counts as detected if the detector fires between its start and
`detection_window_seconds` after its end. Add your own sets (recorded positives, a pause
variant, "hey + other word" phrases, long household recordings) under `streaming_eval.sets`:

```yaml
streaming_eval:
  debounce_seconds: 2.0
  min_consecutive: 1
  target_fa_per_hour: 0.5
  max_clips_per_set: 1000
  sets:
    - {name: positive_pause, positive: true, paths: [./eval/positive_pause]}
    - {name: other_speech, positive: false, paths: [./eval/hey_other_word]}
    - {name: household, positive: false, paths: [./eval/household.wav]}
  # Optional noise bed under every stream, e.g. DEMAND at 10 dB SNR
  background_paths: [./data/demand]
  snr_db: 10.0
```

Paths are WAV files or directories (searched recursively; `clip_NNNNNN.wav` originals are
preferred when present, so augmented copies are skipped).

### Threshold selection

A DET curve (pooled miss rate over positive sets vs pooled false accepts per hour over
negative sets) is computed for thresholds 0.01-0.99. The selected threshold has the lowest miss
rate with at most `target_fa_per_hour` false accepts per hour; if none qualifies, the threshold
with the fewest false accepts is reported with `target_met: false`.

### Output files

| File | Description |
|------|-------------|
| `<model_name>_streaming_eval.json` | Per-set results at the selected (or `-t`) threshold, the selected point, and the DET points |
| `<model_name>_streaming_scores.npz` | Per-hop scores of every set, for re-analysis without re-running the model |
| `<model_name>_streaming_det.png` | Streaming DET curve with the selected threshold marked |
