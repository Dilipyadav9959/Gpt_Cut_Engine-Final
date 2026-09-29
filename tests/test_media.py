"""Real FFmpeg integration checks; no mock renderer or media fixtures."""
import math
import subprocess
import wave

import numpy as np
import pytest

from editron_cut_engine.media import (extract_wav, probe, quantize_keeps, render,
                                     resolve_binary, run, validate_source_timing)


@pytest.fixture(scope="module")
def ffmpeg_tools():
    try:
        binaries = {name: resolve_binary(name) for name in ("ffmpeg", "ffprobe")}
        for binary in binaries.values():
            run([binary, "-version"])
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"Real FFmpeg tools are required: {exc}")
    return binaries


@pytest.fixture(scope="module")
def source(tmp_path_factory, ffmpeg_tools):
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    video = "color=c=black:s=160x90:r=25:d=4,drawbox=color=white:t=fill:enable='between(t,0.24,0.35)+between(t,2.24,2.35)'"
    audio = "aevalsrc='if(between(t,0.24,0.35)+between(t,2.24,2.35),0.6*sin(2*PI*1000*t),0)':s=48000:d=4"
    run([ffmpeg_tools["ffmpeg"], "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", video,
         "-f", "lavfi", "-i", audio, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path])
    return path


def test_frame_rounding_expands_keep_only():
    result = quantize_keeps([(0.013, 0.411), (0.799, 1.113)], 25, 2)
    assert [(x.start, x.end) for x in result] == [(0, .44), (.76, 1.12)]
    assert sum(x.end - x.start for x in result) >= (.411 - .013) + (1.113 - .799)


@pytest.mark.parametrize("keeps", [[(1, .2)], [(-.1, 1)], [(0, math.nan)], [(0, 3)], [(0, 1), (.8, 2)]])
def test_invalid_keeps_rejected(keeps):
    with pytest.raises(ValueError):
        quantize_keeps(keeps, 25, 2)


def test_probe_and_extract(source, tmp_path):
    info = probe(source)
    assert info["duration"] == pytest.approx(4)
    assert (info["width"], info["height"], info["fps"]) == (160, 90, 25)
    wav = tmp_path / "audio.wav"
    extract_wav(source, wav)
    with wave.open(str(wav)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)


def test_cut_render_preserves_sync_landmarks(source, ffmpeg_tools, tmp_path):
    out = tmp_path / "cut.mp4"
    stats = render(source, out, [(0, 1), (2, 4)], preset="ultrafast")
    assert stats["duration"] == pytest.approx(3, abs=.001)
    assert abs(stats["duration_error"]) < .001
    assert abs(stats["audio_video_end_delta"]) <= .04
    assert (stats["width"], stats["height"], stats["fps"]) == (160, 90, 25)
    wav = tmp_path / "rendered.wav"
    extract_wav(out, wav)
    with wave.open(str(wav)) as audio:
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2") / 32768.0
        rate = audio.getframerate()
    # The second audio marker moves from 2.24s to 1.24s after a 1s cut.
    for expected in (.24, 1.24):
        window_start = int((expected - .05) * rate)
        active = np.where(np.abs(samples[window_start:int((expected + .12) * rate)]) > .25)[0]
        assert active.size
        assert (window_start + active[0]) / rate == pytest.approx(expected, abs=.01)
    decoded = subprocess.run([ffmpeg_tools["ffmpeg"], "-v", "error", "-i", str(out),
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(-1, 90, 160)
    bright = np.where(frames.mean(axis=(1, 2)) > 200)[0]
    assert bright.tolist() == [6, 7, 8, 31, 32, 33]


def test_render_no_audio_and_resize(source, ffmpeg_tools, tmp_path):
    silent = tmp_path / "video_only.mp4"
    run([ffmpeg_tools["ffmpeg"], "-v", "error", "-i", source, "-an", "-c:v", "copy", silent])
    assert not probe(silent)["has_audio"]
    stats = render(silent, tmp_path / "vertical.mp4", [(0, 1)], width=180, height=320,
                   fps=30, preset="ultrafast")
    assert (stats["width"], stats["height"], stats["fps"]) == (180, 320, 30)
    assert stats["audio_video_end_delta"] is None


def test_existing_and_source_files_protected(source, tmp_path):
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"keep this file")
    with pytest.raises(FileExistsError):
        render(source, target, [(0, 1)])
    assert target.read_bytes() == b"keep this file"
    with pytest.raises(ValueError, match="different"):
        render(source, source, [(0, 1)], overwrite=True)


def test_render_failure_does_not_publish(source, tmp_path):
    target = tmp_path / "failure.mp4"
    with pytest.raises(RuntimeError):
        render(source, target, [(0, 1)], video_codec="nonexistent_editron_codec")
    assert not target.exists()
    assert not list(tmp_path.glob(".editron-render-*"))


def test_offset_source_rejected(source, ffmpeg_tools, tmp_path):
    offset = tmp_path / "offset.mp4"
    run([ffmpeg_tools["ffmpeg"], "-v", "error", "-i", source, "-c", "copy",
         "-output_ts_offset", "1", offset])
    with pytest.raises(ValueError, match="start time"):
        render(offset, tmp_path / "cut.mp4", [(0, 1)])


def test_vfr_source_rejected(source, ffmpeg_tools, tmp_path):
    vfr = tmp_path / "vfr.mp4"
    run([ffmpeg_tools["ffmpeg"], "-v", "error", "-i", source, "-an", "-vf",
         "select='not(eq(mod(n,3),0))'", "-fps_mode", "vfr", vfr])
    with pytest.raises(ValueError, match="frame rate|frame timestamps|start time"):
        render(vfr, tmp_path / "cut.mp4", [(0, 1)])


def test_invalid_binary_override_is_actionable(monkeypatch):
    monkeypatch.setenv("EDITRON_FFMPEG", "this-executable-does-not-exist-12345")
    with pytest.raises(FileNotFoundError, match="EDITRON_FFMPEG"):
        resolve_binary("ffmpeg")
