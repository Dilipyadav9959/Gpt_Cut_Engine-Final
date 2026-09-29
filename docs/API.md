# API integration

### POST `/v1/cut`
Multipart fields:
- `file`: source video
- `language`: `auto`, `te`, `hi`, `en`, etc.
- `silence_mode`: `vad`, `rms`, `off`
- `repetitions`: `off`, `conservative`, `balanced`, `aggressive`
- `preview`: `true` returns the EDL JSON instead of the rendered file

The JSON EDL is the contract between Editron's caption engine and cut engine. A future main Editron backend can consume `keep_ranges`, `cut_ranges`, `repetition_cuts`, and `words` without depending on the renderer.
