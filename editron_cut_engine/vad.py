"""Audio activity backends. RMS scans PCM in bounded frames."""
import wave
import numpy as np


def read_pcm16(path):
    with wave.open(str(path), "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        if sw != 2:
            raise ValueError("Expected 16-bit PCM WAV")
        data = wf.readframes(wf.getnframes())
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _merge(ranges, max_gap):
    if not ranges:
        return []
    out = [list(ranges[0])]
    for s, e in sorted(ranges)[1:]:
        if s <= out[-1][1] + max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out if b > a]


def rms_speech(path, threshold_db=-38., frame_ms=30, min_speech=.15, min_silence=.25):
    if not np.isfinite(threshold_db) or not -120 <= threshold_db <= 0:
        raise ValueError("RMS threshold must be between -120 and 0 dB")
    if frame_ms <= 0 or min_speech < 0 or min_silence < 0:
        raise ValueError("Invalid RMS time settings")
    ranges = []
    start = None
    with wave.open(str(path), "rb") as wf:
        sr, channels = wf.getframerate(), wf.getnchannels()
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV")
        frame = max(1, round(sr * frame_ms / 1000))
        offset = 0
        while True:
            data = wf.readframes(frame)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.
            samples = samples.reshape(-1, channels)
            # Any channel can carry meaningful speech; don't phase-cancel stereo.
            rms = float(np.sqrt(np.mean(samples*samples, axis=0)).max())
            active = 20*np.log10(max(rms, 1e-12)) >= threshold_db
            if active and start is None:
                start = offset/sr
            if not active and start is not None:
                ranges.append((start, offset/sr))
                start = None
            offset += len(samples)
        if start is not None:
            ranges.append((start, offset/sr))
    return [(s, e) for s, e in _merge(ranges, min_silence) if e-s >= min_speech]


def silero_speech(path, threshold=.5, min_speech=.15, min_silence=.25):
    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps
    except ImportError:
        return None
    x, sr = read_pcm16(path)
    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(x), model, threshold=threshold, sampling_rate=sr,
                              min_speech_duration_ms=int(min_speech*1000),
                              min_silence_duration_ms=int(min_silence*1000))
    return [(t["start"]/sr, t["end"]/sr) for t in ts]
