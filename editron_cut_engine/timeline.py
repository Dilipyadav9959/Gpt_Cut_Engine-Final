"""Interval algebra and strict timeline validation, independent of detectors/rendering."""
import math
from .models import KeepInterval

EPSILON = 1e-6


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_ranges(duration, ranges, *, ordered=True):
    if not _finite(duration) or duration <= 0:
        raise ValueError("Media duration must be finite and positive")
    previous = 0.0
    for start, end in ranges:
        if not _finite(start) or not _finite(end) or not 0 <= start < end <= duration + EPSILON:
            raise ValueError(f"Invalid interval {start!r}..{end!r} for duration {duration}")
        if ordered and start < previous - EPSILON:
            raise ValueError("Intervals must be sorted and non-overlapping")
        previous = end


def merge_intervals(ranges, gap=0.0):
    ranges = sorted((float(a), float(b)) for a, b in ranges if b > a)
    if not ranges:
        return []
    if any(not math.isfinite(a) or not math.isfinite(b) or a < 0 for a, b in ranges):
        raise ValueError("Intervals must be finite and nonnegative")
    out = [list(ranges[0])]
    for a, b in ranges[1:]:
        if a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def subtract_intervals(ranges, blockers):
    result = []
    for start, end in merge_intervals(ranges):
        cursor = start
        for a, b in merge_intervals(blockers):
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                result.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def complement(duration, ranges):
    ranges = merge_intervals(ranges)
    validate_ranges(duration, ranges)
    return subtract_intervals([(0.0, duration)], ranges)


def validate_keeps(duration, keeps):
    pairs = [(k.start, k.end) if hasattr(k, "start") else
             (k["start"], k["end"]) if isinstance(k, dict) else tuple(k) for k in keeps]
    validate_ranges(duration, pairs)
    if not pairs:
        raise ValueError("Timeline must retain some media")
    return pairs


def validate_timeline(duration, timeline):
    if not timeline:
        raise ValueError("Timeline cannot be empty")
    validate_ranges(duration, [(x["start"], x["end"]) for x in timeline])
    cursor = 0.0
    output_cursor = 0.0
    for segment in timeline:
        if abs(segment["start"] - cursor) > EPSILON:
            raise ValueError("Timeline must cover input without gaps")
        if segment.get("action") not in ("KEEP", "CUT"):
            raise ValueError("Only KEEP and CUT are currently executable")
        confidence = segment.get("confidence", 1.0)
        if not _finite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        if not segment.get("reason"):
            raise ValueError("Every interval needs an explainable reason")
        if segment["action"] == "KEEP":
            if "output_start" in segment and abs(segment["output_start"] - output_cursor) > EPSILON:
                raise ValueError("Incorrect output timestamp mapping")
            output_cursor += segment["end"] - segment["start"]
            if "output_end" in segment and abs(segment["output_end"] - output_cursor) > EPSILON:
                raise ValueError("Incorrect output end mapping")
        cursor = segment["end"]
    if abs(cursor - duration) > EPSILON or output_cursor <= 0:
        raise ValueError("Timeline must cover input and retain some media")
    return output_cursor


def build_timeline(duration, accepted):
    """Accepted decisions must already be disjoint, safe, frame aligned cuts."""
    accepted = sorted(accepted, key=lambda x: x["start"])
    validate_ranges(duration, [(x["start"], x["end"]) for x in accepted])
    timeline = []
    cursor = output_cursor = 0.0
    for decision in accepted + [{"start": duration, "end": duration}]:
        if decision["start"] > cursor:
            length = decision["start"] - cursor
            timeline.append(dict(start=cursor, end=decision["start"], action="KEEP",
                                 reason="retained_or_uncertain", confidence=1.0,
                                 output_start=output_cursor, output_end=output_cursor + length))
            output_cursor += length
        if decision["end"] > decision["start"]:
            timeline.append(dict(start=decision["start"], end=decision["end"], action="CUT",
                                 reason=decision["reason"], confidence=decision["confidence"],
                                 decision_id=decision["id"]))
        cursor = decision["end"]
    validate_timeline(duration, timeline)
    return timeline


def build_keep(duration, speech_ranges, cut_intervals, padding=0.08):
    """Legacy low-level helper. New code should use the decision engine."""
    speech = merge_intervals([(max(0, s-padding), min(duration, e+padding))
                              for s, e in speech_ranges])
    validate_ranges(duration, speech)
    cuts = [(c.start, c.end) for c in cut_intervals]
    validate_ranges(duration, merge_intervals(cuts))
    return [KeepInterval(s, e) for s, e in subtract_intervals(speech, cuts)]


def cut_list(duration, keeps):
    return complement(duration, validate_keeps(duration, keeps))
