# Screen Watcher

A local desktop application that periodically observes the screen, detects visual changes, and builds a semantic history of the user's activity. It does not act on the desktop, use cloud services, or store data in a database.

## Journey

The project began as a prototype for benchmarking Qwen3-VL inference on screen captures using DXcam. Tests were run with Qwen3-VL 2B, 4B, and 8B on an NVIDIA RTX 3090; the 2B model provided the most practical initial balance of latency and VRAM usage.

The next iteration split the workflow into small steps:

```text
capture -> inexpensive visual change detector -> Qwen3-VL -> semantic state -> history
```

The detector compares grayscale thumbnails using mean difference and the proportion of changed pixels. Inference is requested only after consecutive relevant changes are confirmed.

## Current status

- Local screen capture with DXcam.
- Local Qwen3-VL inference with CUDA.
- Structured output containing `app`, `activity` category, and a short `summary`.
- Stable categories: `development`, `terminal`, `web`, `media`, `communication`, `document`, `design`, `gaming`, `idle`, and `other`.
- In-memory history that adds events only when `(normalized app, normalized activity)` changes.
- Recovery of JSON wrapped in Markdown blocks and truncated responses when `app` and `activity` are complete.
- Tkinter interface with a capture preview, current state, timeline, and local settings.
- Semantic parsing and comparison tests.

The project does not include a web interface, database, standalone OCR, remote telemetry, or desktop automation.

## Run

Requires Windows, a compatible CUDA GPU, and Python with the dependencies listed in `requirements.txt`.

```powershell
.\.venv\Scripts\python.exe screen_watcher_app.py
```

To run only the watcher from the console:

```powershell
.\.venv\Scripts\python.exe watcher.py
```

## Configuration

`config.json` contains the model, resolution, interval, token limit, and visual detector parameters. The interface exposes the main settings through `Settings`; changes are saved to this file and restart the watcher.

## Project structure

| File | Responsibility |
| --- | --- |
| `screen_watcher_app.py` | Local window and controls. |
| `watcher.py` | Capture, inference, and event worker. |
| `change_detector.py` | Visual change detection without an additional ML model. |
| `activity.py` | Semantic state, normalization, and history. |
| `config.json` | Local configuration. |
| `test_semantic_flow.py` | Semantic-state and parsing tests. |
| `firstTest.py` | Original benchmark and prototype, preserved unchanged. |
