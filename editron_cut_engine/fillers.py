"""Conservative language-scoped hesitation proposals.

Lexicons contain vocalizations only. "Like", "basically", and "you know" need
context and are deliberately not automatic fillers. Unknown language stays safe.
"""

import math

from .models import CutInterval
from .text import token_key


# Candidate spellings, not claims that a vocalization is meaningless.
# "hmm" can mean agreement and is always review-only.
HESITATIONS = {
    "en": {"um", "umm", "uh", "uhh", "erm", "hmm"},
    "te": {"ఉమ్", "ఉమ్మ్", "హ్మ్"},
    "hi": {"उम्", "उम्म", "हम्म"},
    "kn": {"ಉಮ್", "ಉಮ್ಮ್", "ಹ್ಮ್"},
    "ta": {"உம்", "ஹ்ம்ம்"},
    "ml": {"ഉം", "ഹ്മ്"},
}


def detect_fillers(words, language):
    """Return candidates with evidence and word indices for the decision gate.

    Non-English proposals remain review-only until language-specific evaluation.
    Confidence values are conservative heuristics, not calibrated probabilities.
    """
    language = (language or "").lower().split("-")[0]
    lexicon = HESITATIONS.get(language)
    if not lexicon:
        return []
    lexicon = {token_key(token) for token in lexicon}
    candidates = []
    for index, word in enumerate(words):
        token = token_key(word.text)
        if token not in lexicon:
            continue
        if (not all(math.isfinite(value) for value in (word.start, word.end, word.probability))
                or word.start < 0 or word.end <= word.start or not 0 <= word.probability <= 1):
            continue
        before = word.start - words[index - 1].end if index else None
        after = words[index + 1].start - word.end if index + 1 < len(words) else None
        isolated = before is not None and after is not None and before >= 0.12 and after >= 0.12
        strong = (language == "en" and token != "hmm" and isolated
                  and word.probability >= 0.90 and 0.08 <= word.end - word.start <= 0.8)
        candidates.append(CutInterval(
            word.start, word.end, reason="filler_word", confidence=0.94 if strong else 0.60,
            detail=f"Language-scoped hesitation candidate ({language})",
            evidence={
                "language": language, "lexicon_match": True,
                "asr_probability": word.probability, "pause_before": before,
                "pause_after": after, "isolated": isolated,
                "candidate_only": not strong, "transcript": word.text.strip(),
                "ambiguity": None if strong else "context_or_language_requires_review",
            },
            word_indices=[index],
        ))
    return candidates
