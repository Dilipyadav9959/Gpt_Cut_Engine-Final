# Architecture / auto-editor comparison

## Previous auto-editor approach
Auto-Editor's current documentation describes an analysis → timebase labels → actions architecture. Its default edit method is audio loudness: the loudest sample in each timebase section is compared against a normalized amplitude threshold. Active sections are kept and inactive sections are cut. `--margin` expands active regions and `--smooth` suppresses very short cuts/clips. It can also combine audio, motion, black-frame and subtitle/word methods with boolean operators.

## What Editron keeps
- Analysis → intervals → actions architecture.
- Safety margin around speech.
- Minimum-duration filtering / smoothing concept.
- FFmpeg as the media/render layer.
- A JSON Edit Decision List (EDL) as the integration boundary.

## What Editron changes
- Silence decision is speech-aware: Silero VAD is attempted first when installed; RMS is the deterministic fallback.
- Faster-Whisper is used for **word timestamps**, not to decide silence alone.
- Repetition detection works on normalized Unicode tokens, fuzzy phrase similarity, timing and Whisper word confidence.
- The engine keeps the later/corrected phrase and removes the earlier duplicate only when the evidence is strong enough.
- The core is Python and has no dependency on Auto-Editor's Nim runtime, so it is easy to embed in Editron's existing backend.

## Recommended production stack for Editron
`FFmpeg → Silero VAD → Faster-Whisper large-v3 → repetition detector → Edit Decision List → FFmpeg renderer`

For your existing Telugu pipeline, keep `large-v3` + CUDA/float16 on the GPU. The cut engine accepts the same model/device settings you already use.
