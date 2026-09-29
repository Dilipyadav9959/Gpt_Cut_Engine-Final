"""Process adapter for the existing standalone Auto-Editor installation."""
import json
import math
import tempfile
from fractions import Fraction
from pathlib import Path

from .media import run, resolve_binary
from .timeline import merge_intervals, validate_ranges


def parse_v1(data, duration, fps):
    if str(data.get("version")) != "1":
        raise ValueError("Expected Auto-Editor v1 timeline")
    timebase = float(Fraction(str(data.get("timebase", fps))))
    if not math.isfinite(timebase) or timebase <= 0:
        raise ValueError("Invalid Auto-Editor timebase")
    cursor = 0
    keeps = []
    for chunk in data.get("chunks", []):
        if not isinstance(chunk, list) or len(chunk) != 3:
            raise ValueError("Malformed Auto-Editor chunk")
        start, end, speed = chunk
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in chunk):
            raise ValueError("Non-finite Auto-Editor chunk")
        if start != cursor or end <= start or start < 0:
            raise ValueError("Auto-Editor chunks must cover a continuous source timeline")
        if speed not in (0, 1, 99999):
            raise ValueError("Speed changes are not activity ranges")
        if end / timebase > duration + 1 / timebase + 1e-6:
            raise ValueError("Auto-Editor timeline exceeds source duration")
        if speed == 1 and start / timebase < duration:
            keeps.append((start / timebase, min(duration, end / timebase)))
        cursor = end
    if data.get("chunks") and abs(cursor / timebase - duration) > 1 / timebase + 1e-6:
        raise ValueError("Incomplete Auto-Editor source timeline")
    validate_ranges(duration, keeps)
    return merge_intervals(keeps)


def analyze_activity(input_path, duration, fps, *, executable=None, threshold=0.04, margin="0.2s"):
    binary = executable or resolve_binary("auto-editor")
    version = run([str(binary), "--version"]).strip()
    with tempfile.TemporaryDirectory(prefix="editron_auto_editor_") as td:
        timeline_path = Path(td) / "activity.v1"
        command = [str(binary), str(Path(input_path).resolve()), "--edit", f"audio:threshold={threshold}",
                   "--margin", str(margin), "--smooth", "0.2s,0.1s", "--export", "v1",
                   "--output", str(timeline_path), "--no-open", "--no-cache", "--quiet"]
        run(command)
        raw = json.loads(timeline_path.read_text(encoding="utf-8"))
    ranges = parse_v1(raw, duration, fps)
    return ranges, {"backend": "auto-editor", "version": version, "threshold": threshold,
                    "margin": margin, "smooth": "0.2s,0.1s", "timeline": raw}
