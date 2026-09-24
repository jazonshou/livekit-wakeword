# Augmentation Pipeline

The augmentation stage applies realistic audio transformations to synthetic TTS clips and aligns them within detection windows.

**Source:** `src/livekit/wakeword/data/augment.py`
**CLI:** `livekit-wakeword augment <config>`

## Overview

```
Original TTS clips (clip_000000.wav)
    │
    ▼  Round 0: reads originals
    ├──► Per-sample augmentations (EQ, distortion)
    ├──► Alignment (fit to the window)
    ├──► RIR convolution
    ├──► Background mixing (covers the whole window)
    └──► clip_000000_r0.wav
              │
              ▼  Round 1: reads r0 output (stacks)
              ├──► Per-sample augmentations
              ├──► RIR convolution
              ├──► Background mixing
              └──► clip_000000_r1.wav
                        │
                        ▼  ... Round N reads r(N-1)
```

## AudioAugmentor

The `AudioAugmentor` class manages all audio augmentations.

### Initialization

```python
AudioAugmentor(
    background_paths: list[Path],  # Directories with background noise .wav files
    rir_paths: list[Path],         # Directories with room impulse response .wav files
    sample_rate: int = 16000
)
```

All `.wav` files are collected recursively from the provided directories.

### Per-Sample Augmentations

Applied via the `audiomentations` library to individual clips:

| Transform | Probability | Description |
|-----------|------------|-------------|
| `SevenBandParametricEQ` | 0.25 | 7-band parametric equalizer |
| `TanhDistortion` | 0.25 | Tanh-based distortion |

### RIR Convolution

`apply_rir(audio, p=0.5)` convolves audio with a randomly selected room impulse response using FFT convolution (`scipy.signal.fftconvolve`). The RIR is normalized by its maximum absolute value before convolution. Output is cropped to the original audio length. On round 0 the clip is already padded to the window, so the reverb tail rings into the padding after the phrase instead of being cut off.

### Background Mixing

`mix_with_background(audio, snr_db_range=(5.0, 15.0), signal_power=None)` mixes audio with a random background noise clip at a randomly selected SNR within the given range.

The background clip is looped (tiled) if shorter than the audio and randomly cropped to a starting position, so it spans the whole array. The mixing formula scales the background based on:

```
scale = sqrt(signal_power / (background_power * 10^(snr_db / 10)))
output = audio + scale * background
```

`signal_power` defaults to the power of the whole array. On round 0 the pipeline passes the power of just the clip's span within the window: including the zero padding would lower the measured power and make the noise quieter than the requested SNR.

> **Note:** Background noise files serve double duty — they are used here as augmentation overlays *and* also generated as standalone background clips during the [data generation step](data-generation.md#background-noise-clip-generation). Those background clips then pass through this same augmentation pipeline.

## Clip Alignment

On round 0, each clip is fitted to the target window (default 2.0 seconds = 32,000 samples) **before** RIR and background mixing.

### Positive and Negative Clips — End-Aligned

`align_clip_to_end(audio, target_length, jitter_samples=3200)`

Positive and negative clips are both placed at the **end** of the window with random jitter of up to 3200 samples (200ms at 16kHz). Clips longer than the window keep their end. This simulates the real detection scenario where the wake word appears at the trailing edge of the audio buffer.

```
[    padding    |  phrase  | jitter ]
◄──────────── target_length ────────►
```

The padding is zero only until background mixing, which then fills the whole window with noise.

### Background Clips — Center-Padded

Background clips are already window-length when [generated](data-generation.md#background-noise-clip-generation). Any other length is center-cropped or center-padded.

### Why both classes are aligned the same way

Live audio never contains digital silence, and `WakeWordListener` slides a 2 s window in 80 ms hops, so every phrase it hears eventually ends at the edge of the window. If positives sat at the end of the window, negatives sat in the middle, and the padding stayed exact zeros, the classifier could tell the classes apart by where the silence is. Near-miss phrases ("hey Chuck" for "hey Zuck") would then score low on their training clips yet fire when streaming. Aligning both classes the same way and mixing noise over the whole window takes that shortcut away, so the classifier has to learn from the sound of the phrase. openWakeWord uses the same approach.

## Augmentation Rounds

The augmentation pipeline runs `config.augmentation.rounds` passes over all six directories (positive train/test, negative train/test, background train/test). Each round writes to a separate file (`clip_000000_r0.wav`, `clip_000000_r1.wav`, etc.) — originals are never modified.

Rounds **stack**: round 0 reads the clean TTS originals, round 1 reads round 0's output, round 2 reads round 1's output, and so on. This produces progressively more degraded audio as augmentation effects compound across rounds. Old augmented files (`_rN.wav`) are cleaned up at the start of each run so re-running is idempotent.

## Per-Clip Processing Order

For each WAV file in a directory:

1. Read audio, convert to float32, take first channel if stereo
2. Apply per-sample augmentations (EQ, distortion)
3. Align to window — round 0 only (end-aligned for positives and negatives, center-padded for background clips)
4. Apply RIR convolution (50% probability)
5. Mix with background noise across the whole window
6. Write to `clip_NNNNNN_r{round}.wav` (originals preserved)

## Output

After augmentation:

```
output/<model_name>/
├── positive_train/
│   ├── clip_000000.wav             # Original TTS (preserved, not used for training)
│   ├── clip_000000_r0.wav          # Round 0 augmented
│   ├── clip_000000_r1.wav          # Round 1 (stacked on r0)
│   └── ...
├── positive_test/
├── negative_train/
├── negative_test/
├── background_train/
└── background_test/
```

Only `_rN.wav` files are fed to feature extraction — clean TTS originals are excluded from training since they don't match real microphone audio.

Feature extraction is a separate step — see [Feature Extraction](feature-extraction.md).
