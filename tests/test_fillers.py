import pytest

from editron_cut_engine.fillers import detect_fillers
from editron_cut_engine.models import Word


def isolated(token="um", probability=.95):
    return [Word("today", 0, .3, .95), Word(token, .5, .7, probability), Word("discuss", .9, 1.2, .95)]


def test_isolated_high_confidence_hesitation_proposal():
    proposals = detect_fillers(isolated(), "en")
    assert len(proposals) == 1
    assert proposals[0].confidence > .9
    assert proposals[0].word_indices == [1]
    assert proposals[0].evidence["isolated"]


@pytest.mark.parametrize("token", ["basically", "like", "you know", "well", "so"])
def test_meaningful_english_words_are_not_lexical_fillers(token):
    assert not detect_fillers(isolated(token), "en")


@pytest.mark.parametrize("language", [None, "auto", "te", "hi", "kn", "ta", "ml", "fr"])
def test_english_lexicon_never_applied_to_other_language(language):
    assert not detect_fillers(isolated(), language)


def test_low_confidence_hesitation_stays_review_only():
    proposal = detect_fillers(isolated(probability=.4), "en")[0]
    assert proposal.evidence["candidate_only"]
    assert proposal.confidence < .7


def test_connected_hesitation_stays_review_only():
    words = [Word("today", 0, .5, .95), Word("um", .5, .7, .99), Word("discuss", .7, 1, .95)]
    assert detect_fillers(words, "en")[0].evidence["candidate_only"]


@pytest.mark.parametrize("language,token", [("te", "ఉమ్"), ("hi", "उम्म"), ("kn", "ಉಮ್"),
                                           ("ta", "உம்"), ("ml", "ഉം"), ("en", "hmm")])
def test_contextual_or_unvalidated_vocalization_requires_review(language, token):
    assert detect_fillers(isolated(token), language)[0].evidence["candidate_only"]


def test_nan_timestamp_cannot_propose_filler():
    assert not detect_fillers([Word("um", float("nan"), .5, .95)], "en")
