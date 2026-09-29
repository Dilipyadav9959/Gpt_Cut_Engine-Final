# Editron Cut Engine

A merge-friendly video cut engine for **Editron**. It removes dead air and speech repetitions while preserving the intended/latest pronunciation.

## What it takes from auto-editor
Auto-Editor is built around an analysis → timeline labels → actions pipeline. Its default audio method marks timebase sections as active when audio amplitude crosses a threshold, then cuts inactive sections; it also uses margins and smoothing around cuts. This project keeps that architecture but replaces the core speech decision with a more speech-aware hybrid pipeline.

Auto-Editor source is public-domain/Unlicense. This project does **not** copy its implementation; it is a clean Python engine inspired by the documented behavior and compatible conceptually with Editron's pipeline. See `docs/ARCHITECTURE.md`.

## Pipeline
1. FFmpeg extracts mono 16 kHz PCM.
2. **Speech activity**: Silero VAD when installed; RMS fallback otherwise.
3. **Transcription**: Faster-Whisper (`large-v3` recommended for Editron multilingual work) with word timestamps.
4. **Repetition detector**: conservative token/phrase matching using normalized Unicode text, fuzzy similarity, word confidence, timing and restart gaps. It marks the earlier duplicate as removable and keeps the later/corrected occurrence.
5. Timeline builder merges speech and repetition intervals and adds configurable safety padding.
6. FFmpeg renders frame-accurate output.
7. A JSON edit decision list is emitted so the main Editron app can preview, audit, or render through another backend.

## Quick start

### Windows
Install FFmpeg and make sure `ffmpeg.exe` is on PATH. Then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m editron_cut_engine.cli --input input.mp4 --output output.mp4 --language te
```

For GPU Whisper:

```powershell
python -m editron_cut_engine.cli --input input.mp4 --output output.mp4 --language te --whisper-model large-v3 --device cuda --compute-type float16
```

### API

```bash
uvicorn editron_cut_engine.api:app --host 0.0.0.0 --port 8090
```

POST `/v1/cut` as multipart form with `file`, or call the Python service directly. See `docs/API.md`.

## Main CLI options

```text
--silence-mode vad|rms|off
--vad-threshold 0.50
--min-speech 0.15
--min-silence 0.25
--padding 0.08
--repetitions off|conservative|balanced|aggressive
--max-repeat-gap 1.80
--max-phrase-words 6
--whisper-model large-v3
--device auto|cpu|cuda
--compute-type auto|float16|int8_float16|int8
--language auto|te|hi|en|...
--preview-json decision.json
```

## Accuracy note
No automatic editor can promise 100% semantic accuracy. The engine is intentionally conservative: silence is speech-aware first, while repeated-word deletion requires multiple signals. Always keep the EDL JSON and use preview mode for production workflows.
