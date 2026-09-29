"""Language-independent restart proposals with explicit uncertainty."""

import math

from .models import CutInterval
from .text import token_key


def _valid_words(words):
    return [(i, word) for i, word in enumerate(words) if token_key(word.text)]


def detect_repetitions(words, mode="balanced", max_gap=1.8, max_phrase_words=6):
    """Return proposals; central decision policy determines whether to cut.

    Compare adjacent, nonoverlapping, exact phrases only. Single repeated words
    remain low-confidence because "very very" and "had had" can be meaningful.
    Legacy modes remain accepted without weakening semantic safety requirements.
    """
    if mode not in {"off", "conservative", "balanced", "aggressive"}:
        raise ValueError("Unknown repetition mode")
    if not math.isfinite(max_gap) or max_gap < 0:
        raise ValueError("max_gap must be finite and nonnegative")
    if isinstance(max_phrase_words, bool) or not isinstance(max_phrase_words, int) or max_phrase_words < 1:
        raise ValueError("max_phrase_words must be a positive integer")
    if mode == "off" or len(words) < 2:
        return []

    max_words = min(max_phrase_words, 4) if mode == "conservative" else max_phrase_words
    gap_limit = min(max_gap, 1.15) if mode == "conservative" else max_gap
    restart_min = 0.25 if mode == "conservative" else 0.18
    valid = _valid_words(words)
    proposals = []
    reserved = set()
    for size in range(max_words, 0, -1):
        for position in range(len(valid) - 2 * size + 1):
            first = valid[position:position + size]
            later = valid[position + size:position + 2 * size]
            indices = [index for index, _ in first]
            all_indices = indices + [index for index, _ in later]
            if any(index in reserved for index in all_indices):
                continue
            if [token_key(w.text) for _, w in first] != [token_key(w.text) for _, w in later]:
                continue
            sequence = first + later
            if any(not all(math.isfinite(v) for v in (w.start, w.end, w.probability))
                   or w.start < 0 or w.end <= w.start or not 0 <= w.probability <= 1
                   for _, w in sequence):
                continue
            if any(left.end > right.start for (_, left), (_, right) in zip(sequence, sequence[1:])):
                continue
            pause = later[0][1].start - first[-1][1].end
            if not 0 <= pause <= gap_limit:
                continue
            if any(right.start - left.end > gap_limit
                   for phrase in (first, later)
                   for (_, left), (_, right) in zip(phrase, phrase[1:])):
                continue
            first_probability = min(w.probability for _, w in first)
            later_probability = min(w.probability for _, w in later)
            ending = first[-1][1].text.rstrip()
            sentence_complete = (ending.endswith((".", "!", "?", "।", "॥"))
                                 and not ending.endswith(("...", "…")))
            reliable_timing = all(w.end - w.start <= 2.0 for _, w in sequence)
            strong_restart = (
                size >= 2 and pause >= restart_min and not sentence_complete
                and reliable_timing and first_probability >= 0.85
                and later_probability >= 0.85 and later_probability + 0.03 >= first_probability
            )
            ambiguity = None if strong_restart else (
                "single_word_may_be_emphasis_or_grammar" if size == 1
                else "insufficient_restart_evidence"
            )
            proposals.append(CutInterval(
                first[0][1].start, first[-1][1].end,
                reason="repeated_word_or_phrase", confidence=0.94 if strong_restart else 0.60,
                detail=f"Exact adjacent duplicate; retain later phrase at {later[0][1].start:.3f}s",
                evidence={
                    "exact_match": True, "adjacent": True, "phrase_words": size,
                    "restart_pause": pause, "asr_probability": first_probability,
                    "later_probability": later_probability, "candidate_only": not strong_restart,
                    "ambiguity": ambiguity, "sentence_complete": sentence_complete,
                    "transcript": " ".join(w.text.strip() for _, w in first),
                    "retained_word_indices": [index for index, _ in later],
                },
                word_indices=indices,
            ))
            # Keep the later occurrence intact, including triple repetitions.
            reserved.update(all_indices)
    return sorted(proposals, key=lambda candidate: (candidate.start, candidate.end))
