"""Time per-hop wake word inference on the current machine (e.g. a Raspberry Pi).

Feeds 80 ms frames through the streaming detector and reports per-hop latency and
the share of one CPU core it uses. Run it on the target device with the settings
you deploy with (64-bit OS, one ONNX Runtime thread):

    python scripts/benchmark_streaming.py --model hey_zuck.onnx --seconds 120
    python scripts/benchmark_streaming.py --model hey_zuck.onnx --wav household.wav
    python scripts/benchmark_streaming.py --model hey_zuck.onnx --scorer window

Only numpy, onnxruntime and the livekit-wakeword package are needed (``--wav`` also
needs soundfile).
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from livekit.wakeword.inference import model as inference_model

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms
HOP_MS = 1000 * FRAME_SAMPLES / SAMPLE_RATE


class _WindowScorer:
    """predict() on the last 2 s, re-run every hop (the pre-streaming listener path)."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._buf = np.zeros(0, dtype=np.int16)

    def process(self, frame: np.ndarray) -> list[dict[str, float]]:
        self._buf = np.concatenate([self._buf, frame])[-25 * FRAME_SAMPLES :]
        if self._buf.shape[0] < 25 * FRAME_SAMPLES:
            return []
        return [self._model.predict(self._buf)]


def _load_audio(wav: Path | None, seconds: float) -> np.ndarray:
    if wav is None:
        rng = np.random.default_rng(0)
        return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 3000).astype(np.int16)
    import soundfile as sf

    audio, sr = sf.read(str(wav), dtype="int16", always_2d=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{wav} is {sr} Hz; resample to 16 kHz first")
    return np.asarray(audio[:, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, type=Path, help="ONNX classifier")
    parser.add_argument("--wav", type=Path, help="16 kHz WAV to replay (default: noise)")
    parser.add_argument("--seconds", type=float, default=60.0, help="Noise length if no --wav")
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads")
    parser.add_argument(
        "--scorer",
        choices=["streaming", "window"],
        default="streaming",
        help="streaming = StreamingWakeWordModel, window = predict() on a 2 s window",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Hops excluded from timings")
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    model = inference_model.WakeWordModel([args.model], sess_options=opts)
    scorer: Any = _WindowScorer(model)
    used = "window"
    if args.scorer == "streaming":
        try:
            streaming = importlib.import_module("livekit.wakeword.inference.streaming")
        except ImportError:
            print("StreamingWakeWordModel not available; falling back to --scorer window")
        else:
            scorer = streaming.StreamingWakeWordModel(model)
            used = "streaming"

    audio = _load_audio(args.wav, args.seconds)
    n_hops = audio.shape[0] // FRAME_SAMPLES
    if n_hops <= args.warmup:
        raise SystemExit(f"Need more than {args.warmup} hops of audio, got {n_hops}")

    wall = np.zeros(n_hops)
    cpu_start = 0.0
    wall_start = 0.0
    for i in range(n_hops):
        if i == args.warmup:
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
        t0 = time.perf_counter()
        scorer.process(audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
        wall[i] = (time.perf_counter() - t0) * 1000
    cpu_s = time.process_time() - cpu_start
    wall_s = time.perf_counter() - wall_start
    timed = wall[args.warmup :]
    audio_s = timed.shape[0] * HOP_MS / 1000

    arch = platform.architecture()[0]
    print(f"machine     {platform.machine()} ({arch}), {os.cpu_count()} cores")
    print(f"python      {platform.python_version()}, onnxruntime {ort.__version__}")
    print(f"scorer      {used}, {args.threads} intra-op thread(s), {timed.shape[0]} hops timed")
    print(
        "per hop     median {:.2f} ms, p95 {:.2f} ms, p99 {:.2f} ms, max {:.2f} ms".format(
            *np.percentile(timed, [50, 95, 99, 100])
        )
    )
    print(f"real time   {wall_s / audio_s:.1%} of the {HOP_MS:.0f} ms hop budget")
    print(f"cpu         {cpu_s / audio_s:.1%} of one core (process CPU time / audio time)")
    late = int(np.sum(timed > HOP_MS))
    print(f"late hops   {late} ({late / timed.shape[0]:.2%}) took longer than {HOP_MS:.0f} ms")


if __name__ == "__main__":
    main()
