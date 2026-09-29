"""Media I/O only: decisions belong to the engine and a validated timeline."""

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from .models import KeepInterval


def resolve_binary(name):
    """Resolve explicit configuration first, then portable/common installations."""
    key = "EDITRON_" + name.upper().replace("-", "_").removesuffix(".EXE")
    override = os.environ.get(key)
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        found = shutil.which(override)
        if found:
            return found
        raise FileNotFoundError(f"{key} does not identify an executable: {override}")
    found = shutil.which(name)
    if found:
        return found
    executable = name.removesuffix(".exe") + (".exe" if os.name == "nt" else "")
    candidates = [Path.cwd() / ".local" / "bin" / executable,
                  Path.home() / ".local" / "bin" / executable]
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        candidates.append(local / "Microsoft" / "WinGet" / "Links" / executable)
        packages = local / "Microsoft" / "WinGet" / "Packages"
        try:
            candidates.extend(sorted(packages.glob(f"Gyan.FFmpeg_*/ffmpeg-*/bin/{executable}"), reverse=True))
        except OSError:
            pass
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except OSError:
            continue
    raise FileNotFoundError(f"{name} was not found. Install it on PATH or set {key} to its executable.")


def run(cmd, *, timeout=None):
    try:
        p = subprocess.run([str(x) for x in cmd], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except OSError as exc:
        raise RuntimeError(f"Could not run {cmd[0]}: {exc}") from exc
    if p.returncode:
        raise RuntimeError(f"{Path(cmd[0]).name} exited with {p.returncode}: {p.stderr[-5000:]}")
    return p.stdout


def _number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def _rate(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def probe(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input media does not exist: {path}")
    raw = json.loads(run([resolve_binary("ffprobe"), "-v", "error", "-show_format",
                          "-show_streams", "-of", "json", path]))
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = raw.get("format", {})
    # The video is the editing clock. Container duration can include AAC padding.
    dur = _number((video or audio or {}).get("duration"), _number(fmt.get("duration")))
    if dur is None or dur <= 0:
        raise ValueError("Media has no finite positive duration")
    rate = (video or {}).get("avg_frame_rate", "0/0")
    fps = _rate(rate)
    if video and fps <= 0:
        raise ValueError("Video has no usable frame rate")
    nominal = _rate((video or {}).get("r_frame_rate", rate))
    return {"duration": dur, "video": video, "audio": audio,
            "has_video": video is not None, "has_audio": audio is not None,
            "fps": fps, "fps_rational": rate,
            "width": (video or {}).get("width"), "height": (video or {}).get("height"),
            "start_time": _number(fmt.get("start_time"), 0.0),
            "suspected_vfr": bool(video and abs(nominal - fps) > max(0.001, fps * 0.001))}


def duration(path):
    return probe(path)["duration"]


def validate_source_timing(info):
    """Reject clocks this version cannot safely align instead of shifting speech."""
    for kind in ("video", "audio"):
        stream = info.get(kind)
        if stream and abs(_number(stream.get("start_time"), info["start_time"])) > 0.002:
            raise ValueError(f"Nonzero {kind} stream start time is unsupported; normalize timestamps before analysis")
    if info.get("suspected_vfr"):
        raise ValueError("Variable frame rate media is unsupported; normalize to constant frame rate before analysis")


def extract_wav(src, dst):
    info = probe(src)
    validate_source_timing(info)
    if not info["has_audio"]:
        raise ValueError("Input media has no audio stream")
    dst = Path(dst)
    if Path(src).resolve() == dst.resolve():
        raise ValueError("Audio extraction must not overwrite the input")
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([resolve_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
         "-n", "-i", src, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", dst])


def _pair(item):
    if isinstance(item, dict):
        return float(item["start"]), float(item["end"])
    if hasattr(item, "start"):
        return float(item.start), float(item.end)
    return float(item[0]), float(item[1])


def quantize_keeps(keeps, fps, duration):
    """Expand KEEP to frame boundaries; never lose retained speech on rounding."""
    from .timeline import validate_keeps
    pairs = [_pair(k) for k in keeps]
    validate_keeps(duration, [KeepInterval(s, e) for s, e in pairs])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be finite and positive")
    result = []
    for s, e in pairs:
        s = max(0.0, math.floor(s * fps + 1e-7) / fps)
        e = min(duration, math.ceil(e * fps - 1e-7) / fps)
        if result and s <= result[-1].end + 1e-7:
            result[-1].end = max(result[-1].end, e)
        else:
            result.append(KeepInterval(s, e))
    return result


def _verify_cfr(src, info):
    """Stream decoded frame times, catching VFR hidden by average-rate metadata."""
    cmd = [resolve_binary("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(src)]
    step = 1.0 / info["fps"]
    tolerance = max(0.0001, step * 0.001)
    previous = None
    count = 0
    with tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True,
                              encoding="utf-8", errors="replace") as proc:
            try:
                for line in proc.stdout:
                    value = _number(line.split(",", 1)[0].strip())
                    if value is None:
                        continue
                    if previous is not None and abs(value - previous - step) > tolerance:
                        raise ValueError("Variable/discontinuous frame timestamps are unsupported; normalize to constant frame rate first")
                    previous = value
                    count += 1
                if proc.wait() != 0:
                    stderr.seek(0)
                    raise RuntimeError("Frame timestamp validation failed: " + stderr.read().decode("utf-8", "replace")[-2000:])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    if count == 0:
        raise ValueError("Video contains no decodable frames")


def _stream_end(stream):
    if not stream:
        return None
    length = _number(stream.get("duration"))
    return None if length is None else _number(stream.get("start_time"), 0.0) + length


def render(src, out, keeps, video_codec="libx264", crf=18, audio_codec="aac", *,
           overwrite=False, width=None, height=None, fps=None, preset="medium"):
    """Render a supplied KEEP list atomically and verify its duration and sync."""
    started = time.perf_counter()
    src, out = Path(src).resolve(), Path(out).resolve()
    if src == out:
        raise ValueError("Output must be different from the input media")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out}; select a new name or explicitly enable overwrite")
    if not out.suffix:
        raise ValueError("Output requires a media extension, such as .mp4")
    if not isinstance(crf, (float, int)) or not math.isfinite(crf) or not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51")
    if (width is None) != (height is None):
        raise ValueError("Set both output width and height, or neither")
    if width is not None and (not isinstance(width, int) or not isinstance(height, int)
                              or width <= 0 or height <= 0 or width % 2 or height % 2):
        raise ValueError("Output width and height must be positive even integers")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("Output FPS must be finite and positive")
    info = probe(src)
    if not info["has_video"]:
        raise ValueError("Rendering requires a video stream")
    validate_source_timing(info)
    effective = quantize_keeps(list(keeps), info["fps"], info["duration"])
    if not effective:
        raise ValueError("No media remains after analysis")
    _verify_cfr(src, info)
    target_fps = fps or info["fps"]
    expected = sum(k.end - k.start for k in effective)
    parts = []
    for i, k in enumerate(effective):
        first, last = round(k.start * info["fps"]), math.ceil(k.end * info["fps"] - 1e-7)
        parts.append(f"[0:v:0]trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS[v{i}]")
        if info["has_audio"]:
            parts.append(f"[0:a:0]atrim=start={k.start:.9f}:end={k.end:.9f},asetpts=PTS-STARTPTS[a{i}]")
    labels = "".join(f"[v{i}]" + (f"[a{i}]" if info["has_audio"] else "") for i in range(len(effective)))
    parts.append(labels + f"concat=n={len(effective)}:v=1:a={int(info['has_audio'])}[joinedv]" + ("[outa]" if info["has_audio"] else ""))
    transforms = []
    if width is not None:
        transforms.extend([f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2",
                           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2", "setsar=1"])
    if fps is not None:
        transforms.append(f"fps={fps:.9f}")
    transforms.append("format=yuv420p")
    parts.append("[joinedv]" + ",".join(transforms) + "[outv]")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Both temporary artifacts live beside the output; final promotion is atomic.
    with tempfile.TemporaryDirectory(prefix=".editron-render-", dir=out.parent) as td:
        graph = Path(td) / "filters.txt"
        temporary = Path(td) / ("render" + out.suffix)
        graph.write_text(";\n".join(parts), encoding="utf-8")
        cmd = [resolve_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-i", src, "-filter_complex_script", graph, "-map", "[outv]", "-c:v", video_codec]
        if video_codec in {"libx264", "libx265"}:
            cmd.extend(["-preset", preset, "-crf", str(crf)])
        if info["has_audio"]:
            cmd.extend(["-map", "[outa]", "-c:a", audio_codec, "-b:a", "192k"])
        cmd.extend(["-fps_mode", "passthrough", "-map_metadata", "0"])
        if out.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(temporary)
        run(cmd)
        actual = probe(temporary)
        duration_error = actual["duration"] - expected
        video_end, audio_end = _stream_end(actual["video"]), _stream_end(actual["audio"])
        sync_delta = audio_end - video_end if audio_end is not None and video_end is not None else None
        tolerance = max(2 / target_fps, 0.05)
        if abs(duration_error) > tolerance:
            raise RuntimeError(f"Rendered duration differs from timeline by {duration_error:.6f}s (limit {tolerance:.6f}s)")
        if info["has_audio"] and (not actual["has_audio"] or sync_delta is None):
            raise RuntimeError("Rendered audio stream is missing or cannot be verified")
        if sync_delta is not None and abs(sync_delta) > tolerance:
            raise RuntimeError(f"Rendered audio/video end times differ by {sync_delta:.6f}s")
        if overwrite:
            os.replace(temporary, out)
        elif os.name == "nt":
            os.rename(temporary, out)
        else:
            os.link(temporary, out)
            temporary.unlink()
    return {"output": str(out), "expected_duration": expected, "duration": actual["duration"],
            "duration_error": duration_error, "audio_video_end_delta": sync_delta,
            "render_seconds": time.perf_counter() - started,
            "effective_keeps": [k.to_dict() for k in effective],
            "width": actual["width"], "height": actual["height"], "fps": actual["fps"],
            "sync_validation": "stream endpoints; perceptual lip sync requires human review"}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    fd, name = tempfile.mkstemp(prefix=".editron-json-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

