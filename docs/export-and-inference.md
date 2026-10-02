# Export & Inference

The export stage converts the trained PyTorch classifier to ONNX or TFLite for deployment. The inference API provides `WakeWordModel` for prediction and `WakeWordListener` for async microphone detection.

**Source:** `src/livekit/wakeword/export/onnx.py`, `src/livekit/wakeword/export/tflite.py`, `src/livekit/wakeword/inference/model.py`, `src/livekit/wakeword/inference/listener.py`
**CLI:** `livekit-wakeword export <config> [--format onnx|tflite]`

The output format is chosen by (in priority order) the `--format` flag, then the `output_format` field in the config (defaults to `onnx`).

## ONNX Export

### Classifier Export

`export_onnx()` exports the trained PyTorch classifier head to ONNX format.

| Property | Value |
|----------|-------|
| Input name | `embeddings` |
| Input shape | `(1, 16, 96)` with dynamic batch axis |
| Output name | `score` |
| Output shape | `(1, 1)` with dynamic batch axis |
| Opset version | 18 |

### INT8 Quantization

`quantize_onnx()` applies dynamic INT8 quantization using `onnxruntime.quantization`:

- Weight type: `QuantType.QInt8`
- Output filename: `<model_name>.int8.onnx`

Enable via the `--quantize` flag:

```bash
livekit-wakeword export configs/hey_jarvis.yaml --quantize
```

### Export Entry Point

`run_export(config, quantize=False, format=None)` loads the trained model from `output/<model_name>/<model_name>.pt`, exports it to ONNX, and optionally quantizes it. `format` defaults to `config.output_format`. ONNX is always produced (it is the conversion source for TFLite); when `format="tflite"`, the TFLite artifact is produced as well and its path is returned. Raises `FileNotFoundError` if the trained model doesn't exist.

## TFLite Export (openWakeWord-compatible)

`export_tflite()` converts an exported ONNX classifier to TFLite via `onnx2tf` (ONNX → TF SavedModel → TFLite), producing an artifact that [openWakeWord](https://github.com/dscripka/openWakeWord) can load directly.

Requires the optional extra:

```bash
uv sync --extra tflite     # or: pip install 'livekit-wakeword[tflite]'
```

```bash
livekit-wakeword export configs/hey_jarvis.yaml --format tflite
```

### openWakeWord contract

openWakeWord loads classifier models with `ai_edge_litert.interpreter` and runs them without resizing tensors, so the artifact must satisfy:

| Requirement | Detail |
|-------------|--------|
| Input shape | **Static** `(1, 16, 96)` float32 (no dynamic batch — openWakeWord never calls `resize_tensor_input`) |
| Output shape | `(1, 1)` float32 sigmoid score |
| Ops | **Builtin TFLite ops only** — the LiteRT interpreter has no Flex/SELECT_TF delegate |

We pin the input shape with onnx2tf's `overwrite_input_shape` + `keep_shape_absolutely_input_names` (without the latter, onnx2tf's NCHW→NHWC pass transposes the input to `(1, 96, 16)`) and restrict the converter to `TFLITE_BUILTINS`.

### Head support

| Head | TFLite export | Notes |
|------|---------------|-------|
| `dnn` | Supported | Bit-exact vs ONNX/PyTorch (verified, maxdiff `0.0`) |
| `conv_attention` | Not supported | onnx2tf emits an unsupported constant for the attention block |
| `rnn` | Not supported | LSTM lowers to `TensorList` ops requiring the Flex delegate (which openWakeWord can't load) |

Use `dnn` for openWakeWord-compatible TFLite; deploy `conv_attention`/`rnn` via ONNX. Requesting TFLite for an unsupported head raises `NotImplementedError` before any export work begins.

## Inference API

**Source:** `src/livekit/wakeword/inference/model.py`, `src/livekit/wakeword/inference/streaming.py`, `src/livekit/wakeword/inference/listener.py`

### WakeWordModel

The `WakeWordModel` class is a stateless prediction API for wake word detection. Pass a complete audio window (~2 seconds) and receive confidence scores.

```python
from livekit.wakeword import WakeWordModel

model = WakeWordModel(models=["hey_livekit.onnx"])

# Pass ~2 seconds of 16kHz audio
scores = model.predict(audio_chunk)
# Returns: {"hey_livekit": 0.95}
```

#### Initialization

```python
WakeWordModel(
    models: list[str | Path] | None = None,  # Paths to ONNX classifiers
)
```

Feature extraction models (`melspectrogram.onnx`, `embedding_model.onnx`) are bundled with the package and loaded automatically.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(audio_chunk)` | `dict[str, float]` | Scores for each loaded model (0-1) |
| `load_model(path, name)` | `None` | Load additional wake word model |

#### Audio Input

- **Format:** 16kHz mono, int16 or float32
- **Chunk size:** ~2 seconds (32,000 samples) recommended — yields 16 embeddings for the classifier
- **Stateless:** No internal audio buffering; the caller manages the audio window

### StreamingWakeWordModel

For a continuous stream, `StreamingWakeWordModel` wraps a `WakeWordModel` and scores every 80 ms hop incrementally. It computes the mel frames of each new hop and one new speech embedding, and keeps the last 16 embeddings, so each hop costs one embedding-model call instead of the 16 that `predict()` makes on a 2 s window (about 5 ms instead of 40 ms per hop on an x86 core).

```python
from livekit.wakeword import StreamingWakeWordModel, WakeWordModel

stream = StreamingWakeWordModel(WakeWordModel(models=["hey_livekit.onnx"]))

for frame in frames:                 # 16 kHz int16 or float32, any length
    for scores in stream.process(frame):   # one dict per completed 80 ms hop
        if scores["hey_livekit"] >= 0.5:
            ...

stream.reset()                       # drop buffered audio, e.g. after a detection
```

| Method | Returns | Description |
|--------|---------|-------------|
| `process(audio)` | `list[dict[str, float]]` | Scores for each 80 ms hop the audio completes; empty during the first 2 s after construction or `reset()` |
| `reset()` | `None` | Drop buffered audio and cached features |

At every hop the scores equal `predict()` on the most recent 2 s of audio, up to float rounding (`tests/test_streaming.py` checks this on every hop). The mel model floors quiet frames at 80 dB below the loudest frame in its input, so when the loudest frame in the 2 s window changes, cached embeddings that touched the floor are recomputed. With a live microphone's noise floor this rarely happens. Several streams can share one `WakeWordModel` and its ONNX sessions.

### WakeWordListener

The `WakeWordListener` class provides async microphone detection with debouncing. It feeds each 80 ms microphone frame to a `StreamingWakeWordModel`.

```python
import asyncio
from livekit.wakeword import WakeWordModel, WakeWordListener

model = WakeWordModel(models=["hey_livekit.onnx"])

async def main():
    async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
        while True:
            detection = await listener.wait_for_detection()
            print(f"Detected {detection.name}! ({detection.confidence:.2f})")

asyncio.run(main())
```

#### Initialization

```python
WakeWordListener(
    model: WakeWordModel | StreamingWakeWordModel,  # loaded classifiers
    threshold: float = 0.5,  # Detection threshold (0-1)
    debounce: float = 2.0,   # Minimum seconds between detections
    min_consecutive: int = 1 # 80 ms frames in a row a score must reach threshold to fire
)
```

#### Detection Result

```python
@dataclass
class Detection:
    name: str        # Model name that triggered
    confidence: float  # Score (0-1)
    timestamp: float   # Monotonic time
```

#### Lifecycle

The listener is designed as an async context manager. On each `__aenter__`, all internal state is reset — including the audio buffer, error state, and detection queue — so the same listener instance can be safely reused across multiple `async with` blocks without stale detections carrying over.

#### Audio Capture

Uses PyAudio to capture from the default microphone:

| Parameter | Value |
|-----------|-------|
| Format | int16 (paInt16) |
| Channels | 1 (mono) |
| Sample rate | 16,000 Hz |
| Buffer size | 1,280 samples (80ms) |
