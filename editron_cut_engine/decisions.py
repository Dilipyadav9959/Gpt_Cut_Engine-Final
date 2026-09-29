"""Conservative, explainable fusion of activity, word timing and local candidates.

Scores are heuristic evidence strengths, not calibrated probabilities of correctness.
"""
import math
from .models import CutInterval
from .timeline import build_timeline, complement, merge_intervals, subtract_intervals, validate_ranges


def overlap(a, b, c, d):
    return min(b, d) - max(a, c) > 1e-7


def validate_words(words, duration):
    last_start = -1.0
    for word in words:
        values = (word.start, word.end, word.probability)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError("Transcript contains non-finite word timestamps/confidence")
        if not 0 <= word.start <= word.end <= duration + 1e-6 or word.start < last_start:
            raise ValueError("Transcript words must be source-relative, ordered and inside media")
        if not isinstance(word.text, str) or not 0 <= word.probability <= 1:
            raise ValueError("Invalid word text/confidence")
        last_start = word.start


def decide(duration, speech_ranges, words, candidates=(), *, fps=0, padding=.08,
           min_pause=.8, min_cut=.2, protected_ranges=(), unaligned_ranges=(),
           secondary_speech=None, mode="smart", min_confidence=.90):
    """Return a full timeline and all accepted/rejected candidates.

    Silence cuts shrink away from protected speech. Semantic cuts must remove complete
    selected words and protect all neighboring words; unsafe candidates stay in review.
    """
    validate_words(words, duration)
    validate_ranges(duration, speech_ranges)
    user_blocks = [(x["start"], x["end"]) if isinstance(x, dict) else tuple(x)
                   for x in protected_ranges]
    validate_ranges(duration, user_blocks, ordered=False)
    validate_ranges(duration, unaligned_ranges, ordered=False)
    blocks = merge_intervals(user_blocks + list(unaligned_ranges))
    word_blocks = [(max(0, w.start-padding), min(duration, w.end+padding)) for w in words]
    silence = complement(duration, speech_ranges)
    proposals = [CutInterval(s, e, "long_pause", .95 if mode == "smart" else .80,
                            evidence={"audio_inactive": True, "pause_duration": e-s})
                 for s, e in silence]
    proposals.extend(candidates)
    decisions = []
    accepted = []
    for index, candidate in enumerate(proposals):
        d = candidate.to_dict()
        d.update(id=f"edit-{index+1:04d}", proposed_start=candidate.start,
                 proposed_end=candidate.end, action="KEEP", status="rejected",
                 confidence_kind="heuristic", transcript=" ".join(
                     words[i].text.strip() for i in candidate.word_indices if 0 <= i < len(words)))
        reason = None
        parts = []
        is_silence = candidate.reason == "long_pause"
        if not (math.isfinite(candidate.confidence) and 0 <= candidate.confidence <= 1):
            raise ValueError("Candidate confidence must be finite and in [0,1]")
        validate_ranges(duration, [(candidate.start, candidate.end)])
        if is_silence:
            if candidate.end-candidate.start < min_pause:
                reason = "short_natural_pause"
            elif mode == "smart" and not words:
                reason = "no_aligned_words_uncertain"
            else:
                # Retain breathing room on both sides, including the source edges.
                start = candidate.start + (padding if candidate.start > 0 else 0)
                end = candidate.end - (padding if candidate.end < duration else 0)
                extra = [] if secondary_speech is None else [
                    (max(0, s-padding), min(duration, e+padding)) for s, e in secondary_speech]
                parts = subtract_intervals([(start, end)], blocks + word_blocks + extra)
                d["evidence"]["word_boundary_protection"] = True
                d["evidence"]["secondary_audio_checked"] = secondary_speech is not None
                # A word inside a silent proposal can split it into short natural pauses.
                parts = [(s, e) for s, e in parts if e-s >= max(min_cut, min_pause-2*padding)]
                if not parts:
                    reason = "protected_speech_pause_or_emphasis"
        elif mode != "smart":
            reason = "semantic_edits_disabled_in_fast_mode"
        elif candidate.confidence < min_confidence:
            reason = "insufficient_evidence"
        elif candidate.evidence.get("candidate_only"):
            reason = "requires_semantic_review"
        elif not candidate.word_indices:
            reason = "no_word_alignment"
        else:
            indices = set(candidate.word_indices)
            if any(i < 0 or i >= len(words) for i in indices):
                raise ValueError("Candidate refers to an unknown word")
            selected = [words[i] for i in sorted(indices)]
            if sorted(indices) != list(range(min(indices), max(indices)+1)):
                reason = "non_contiguous_words"
            elif candidate.start > selected[0].start or candidate.end < selected[-1].end:
                reason = "candidate_clips_selected_word"
            else:
                # Remove entire phonetic event; only expand inside safe neighbor gaps.
                start = max(0., candidate.start-.02)
                end = min(duration, candidate.end+.02)
                if fps:
                    start = max(0., math.floor(start*fps+1e-8)/fps)
                    end = min(duration, math.ceil(end*fps-1e-8)/fps)
                neighbor_blocks = [b for i, b in enumerate(word_blocks) if i not in indices]
                if any(overlap(start, end, s, e) for s, e in blocks + neighbor_blocks):
                    reason = "protected_or_neighboring_word"
                elif not any(overlap(start, end, s, e) for s, e in speech_ranges):
                    reason = "no_audio_corroboration"
                else:
                    parts = [(start, end)]
                    d["evidence"]["audio_corroborated"] = True
                    d["evidence"]["neighbor_words_protected"] = True
        safe_parts = []
        for s, e in parts:
            if is_silence and fps:
                s, e = math.ceil(s*fps-1e-8)/fps, math.floor(e*fps+1e-8)/fps
            if e-s < min_cut-1e-8:
                continue
            if any(overlap(s, e, x["start"], x["end"]) for x in accepted):
                continue
            if is_silence and any(overlap(s, e, w.start, w.end) for w in words):
                raise ValueError("Internal error: silence proposal overlaps a word")
            trial = [(x["start"], x["end"]) for x in accepted] + safe_parts + [(s, e)]
            remaining = complement(duration, trial)
            # Do not delete an all-silent recording, nor strand tiny retained fragments.
            if not remaining or any(b-a < .12-1e-8 for a, b in remaining):
                continue
            safe_parts.append((s, e))
        if safe_parts:
            d.update(status="accepted", action="CUT", decision_reason="multiple_safety_checks_passed",
                     ranges=[{"start":s, "end":e} for s, e in safe_parts])
            for s, e in safe_parts:
                accepted.append(dict(id=d["id"], start=s, end=e, reason=candidate.reason,
                                     confidence=candidate.confidence))
        else:
            d.update(decision_reason=reason or "too_short_overlapping_or_empty_result", ranges=[])
        decisions.append(d)
    return build_timeline(duration, accepted), decisions
