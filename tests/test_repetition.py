import pytest

from editron_cut_engine.models import Word
from editron_cut_engine.repetition import detect_repetitions


def test_repeated_word_is_review_candidate():
    words = [Word("I", 0, .2, .80), Word("I", .35, .55, .95), Word("think", .6, 1, .95)]
    cuts = detect_repetitions(words, "balanced")
    assert cuts and cuts[0].start == 0
    assert cuts[0].confidence < .7
    assert cuts[0].evidence["candidate_only"]
    assert cuts[0].word_indices == [0]


def test_legit_repetition_far_apart():
    words = [Word("very", 0, .2, .95), Word("very", 2.5, 2.7, .95), Word("good", 2.8, 3.1, .95)]
    assert not detect_repetitions(words, "balanced")


@pytest.mark.parametrize("token", ["very", "had", "no", "చాలా", "बहुत"])
def test_emphatic_or_grammatical_single_word_requires_review(token):
    words = [Word(token, 0, .2, .99), Word(token, .5, .7, .99)]
    proposal = detect_repetitions(words)[0]
    assert proposal.confidence < .7
    assert proposal.evidence["candidate_only"]


def test_exact_multiword_restart_has_multiple_signals():
    words = [Word("I", 0, .2, .95), Word("think", .25, .5, .95),
             Word("I", .8, 1, .96), Word("think", 1.05, 1.3, .96)]
    proposals = detect_repetitions(words)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.word_indices == [0, 1]
    assert proposal.evidence["retained_word_indices"] == [2, 3]
    assert proposal.confidence > .9
    assert not proposal.evidence["candidate_only"]


@pytest.mark.parametrize("phrase", [("we", "can", "we", "cannot"),
                                     ("I", "like", "I", "dislike")])
def test_never_skip_intervening_lexical_words_or_fuzzy_match(phrase):
    words = [Word(token, index * .4, index * .4 + .2, .99) for index, token in enumerate(phrase)]
    assert detect_repetitions(words) == []


def test_complete_sentence_repetition_is_ambiguous():
    words = [Word("I", 0, .2, .95), Word("agree.", .25, .5, .95),
             Word("I", .8, 1, .96), Word("agree.", 1.05, 1.3, .96)]
    assert detect_repetitions(words)[0].evidence["candidate_only"]


def test_low_asr_confidence_cannot_be_overruled_by_exact_text():
    words = [Word("I", 0, .2, .4), Word("think", .25, .5, .95),
             Word("I", .8, 1, .96), Word("think", 1.05, 1.3, .96)]
    assert detect_repetitions(words)[0].confidence < .7


def test_ellipsis_can_indicate_an_incomplete_restart():
    words = [Word("I", 0, .2, .95), Word("think...", .25, .5, .95),
             Word("I", .8, 1, .96), Word("think", 1.05, 1.3, .96)]
    assert not detect_repetitions(words)[0].evidence["candidate_only"]


def test_triple_repeat_keeps_retained_occurrence():
    words = [Word("go", index * .4, index * .4 + .2, .99) for index in range(3)]
    proposals = detect_repetitions(words)
    assert len(proposals) == 1
    assert proposals[0].word_indices == [0]
    assert proposals[0].evidence["retained_word_indices"] == [1]


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (.1, .05), (-.1, .1), (.1, .1)])
def test_invalid_timing_cannot_propose_repetition(start, end):
    assert not detect_repetitions([Word("I", start, end, .95), Word("I", .4, .6, .95)])


def test_indic_vowel_difference_is_not_repetition():
    assert not detect_repetitions([Word("కలం", 0, .3, .99), Word("కాలం", .5, .8, .99)])
    assert not detect_repetitions([Word("का", 0, .3, .99), Word("की", .5, .8, .99)])
