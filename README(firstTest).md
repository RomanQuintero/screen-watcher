# VLM Benchmark — Screen Watcher

`firstTest.py` is the initial experiment behind **Screen Watcher**.

Before building the continuous watcher, this script was used to answer a simpler question:

> Can a local Vision-Language Model understand what is happening on the screen fast enough to be useful for continuous monitoring?

The experiment evolved into a small benchmark comparing different Qwen3-VL model sizes, image resolutions and generation limits on the same machine.

## Hardware

- GPU: NVIDIA GeForce RTX 3090 — 24 GB VRAM
- PyTorch: 2.14.0+cu132
- CUDA runtime: 13.2
- Python: 3.13
- Inference: local, using PyTorch + Hugging Face Transformers

No remote inference API is used.

## Pipeline

```text
DXcam
  ↓
Screen capture
  ↓
Resize
  ↓
PIL / Qwen processor
  ↓
Qwen3-VL
  ↓
CUDA / RTX 3090
  ↓
Short semantic description
```

The benchmark measures the complete VLM inference stage rather than screen capture FPS.

## Usage

Run a single inference on the current screen:

powershell

python firstTest.py

Run the VLM benchmark:

python firstTest.py --benchmark

Specify the number of frames:

python firstTest.py --benchmark --frames 10

> The default configuration uses Qwen3-VL-2B at 1024×576 with a short generation budget.

## Experiments

### Qwen3-VL-8B — Native 4K

Initial test using the full 3840×2160 screen.

| Metric | Result |
|---|---:|
| Resolution | 3840×2160 |
| Frames | 5 |
| Total time | 71.60 s |
| Average | 14.31 s/frame |
| Throughput | 0.070 frames/s |
| Minimum | 13.91 s |
| Maximum | 14.57 s |

The model produced detailed and accurate descriptions, correctly recognizing elements such as Visual Studio Code, Python code, terminals, chat interfaces and the file explorer.

However, ~14 seconds per observation was clearly too slow for the intended watcher.

---

### Qwen3-VL-8B — 1024×576

Reducing the input resolution produced a large improvement.

| Metric | Result |
|---|---:|
| Resolution | 1024×576 |
| Frames | 5 |
| Total time | 22.44 s |
| Average | 4.48 s/frame |
| Throughput | 0.223 frames/s |
| Minimum | 3.82 s |
| Maximum | 5.48 s |

This showed that native 4K input was unnecessary for general semantic screen understanding.

---

### Qwen3-VL-8B — 1024×576, 64 output tokens

Generation was then restricted and inference executed without gradient tracking.

| Metric | Result |
|---|---:|
| Resolution | 1024×576 |
| `max_new_tokens` | 64 |
| Total time | 10.93 s |
| Average | 2.18 s/frame |
| Throughput | 0.458 frames/s |
| Minimum | 1.72 s |
| Maximum | 3.21 s |

This demonstrated that output generation was also a significant part of the latency.

---

### Qwen3-VL-8B — 1024×576, 32 output tokens

| Metric | Result |
|---|---:|
| Resolution | 1024×576 |
| `max_new_tokens` | 32 |
| Total time | 7.25 s |
| Average | 1.44 s/frame |
| Throughput | 0.690 frames/s |
| Minimum | 1.30 s |
| Maximum | 1.97 s |

Short outputs were sufficient to identify the main activity on screen, although descriptions could become truncated.

---

### Qwen3-VL-4B — 1024×576, 32 output tokens

| Metric | Result |
|---|---:|
| Resolution | 1024×576 |
| `max_new_tokens` | 32 |
| Total time | 6.24 s |
| Average | 1.24 s/frame |
| Throughput | 0.802 frames/s |
| Minimum | 1.05 s |
| Maximum | 1.88 s |
| VRAM | < 11 GB |

The 4B model reduced latency and memory consumption while retaining useful screen understanding.

Some descriptions, however, became more generic or occasionally inaccurate.

---

### Qwen3-VL-2B — 1024×576, 32 output tokens

| Metric | Result |
|---|---:|
| Resolution | 1024×576 |
| `max_new_tokens` | 32 |
| Total time | 4.81 s |
| Average | 0.95 s/frame |
| Throughput | 1.039 frames/s |
| Minimum | 0.80 s |
| Maximum | 1.44 s |
| VRAM | ~5.8 GB |

This became the most interesting configuration for the application.

Despite being considerably smaller, the model was still able to correctly understand activities such as running a Python performance benchmark.

The lower memory footprint also makes this configuration much more practical than the original 8B experiment.

---

### Qwen3-VL-2B — 1920×1080

A higher-resolution test was also performed to evaluate the trade-off between speed and visual detail.

| Metric | Result |
|---|---:|
| Resolution | 1920×1080 |
| Total time | 10.76 s |
| Average | 2.15 s/frame |
| Throughput | 0.465 frames/s |
| Minimum | 1.08 s |
| Maximum | 3.48 s |

At 1080p the model was noticeably slower, but it could read small UI elements and benchmark values more reliably.

This suggests two useful operating regimes:

- **Lower resolution:** faster semantic perception.
- **Higher resolution:** better UI/text detail.

## Summary

| Model | Resolution | Tokens | Avg. latency | Throughput | VRAM |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 3840×2160 | — | 14.31 s | 0.070 fps | — |
| Qwen3-VL-8B | 1024×576 | — | 4.48 s | 0.223 fps | — |
| Qwen3-VL-8B | 1024×576 | 64 | 2.18 s | 0.458 fps | — |
| Qwen3-VL-8B | 1024×576 | 32 | 1.44 s | 0.690 fps | — |
| Qwen3-VL-4B | 1024×576 | 32 | 1.24 s | 0.802 fps | <11 GB |
| **Qwen3-VL-2B** | **1024×576** | **32** | **0.95 s** | **1.039 fps** | **~5.8 GB** |
| Qwen3-VL-2B | 1920×1080 | — | 2.15 s | 0.465 fps | — |

These results are measurements from this development machine and should not be interpreted as general model benchmarks.

## What this experiment changed

The main conclusion was that continuously running a large VLM over full-resolution screenshots was unnecessary.

Screen Watcher therefore evolved towards a different architecture:

```text
Screen
  ↓
Cheap visual change detector
  ↓
Significant change?
  ├── No  → skip inference
  └── Yes → Qwen3-VL
                ↓
          semantic state
                ↓
          activity history
```

Instead of maximizing raw VLM inference frequency, the application tries to minimize unnecessary inference.

The benchmark also motivated the current use of **Qwen3-VL-2B** as the default model: it provides a useful compromise between semantic understanding, latency and VRAM usage.

## Notes

`firstTest.py` is intentionally kept separate from the main Screen Watcher application.

It represents the experimental stage used to validate the core idea and provides a reproducible way to test different models, resolutions and generation parameters without involving the temporal activity-tracking logic.

## Future work

- Improve temporal semantic tracking.
- Distinguish activity changes from context changes.
- Reduce visual prompt contamination / self-observation.
- Evaluate adaptive resolution depending on screen content.
- Benchmark additional VLMs and quantized variants.
- Add optional object detection and visual overlays.