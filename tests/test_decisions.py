import math
import random
import pytest
from editron_cut_engine.decisions import decide
from editron_cut_engine.models import Word
from editron_cut_engine.fillers import detect_fillers
from editron_cut_engine.repetition import detect_repetitions
from editron_cut_engine.timeline import validate_timeline, merge_intervals, complement


def cuts(timeline):
    return [(x['start'],x['end']) for x in timeline if x['action']=='CUT']


def test_continuous_talking_stays():
    words=[Word('speech',0,4,.99)]
    timeline,decisions=decide(4,[(0,4)],words)
    assert not cuts(timeline)
    assert validate_timeline(4,timeline)==4


def test_long_silence_cut_and_words_preserved():
    words=[Word('hello',0,1,.99),Word('world',4,5,.99)]
    timeline,decisions=decide(5,[(0,1),(4,5)],words,fps=25)
    assert cuts(timeline)==[(1.08,3.92)]
    assert decisions[0]['status']=='accepted'
    assert decisions[0]['reason']=='long_pause'


def test_short_pause_keep():
    timeline,decisions=decide(3,[(0,1),(1.4,3)],[Word('a',0,1,.95),Word('b',1.4,3,.95)])
    assert not cuts(timeline)
    assert decisions[0]['decision_reason']=='short_natural_pause'


def test_emphasis_override_keep():
    timeline,decisions=decide(5,[(0,1),(4,5)],[Word('a',0,1,.9),Word('b',4,5,.9)],
                             protected_ranges=[{'start':1,'end':4,'reason':'emphasis'}])
    assert not cuts(timeline)


def test_whisper_overrides_silence_and_low_confidence_speech_stays():
    timeline,_=decide(5,[(0,1),(4,5)],[Word('quiet Telugu',.9,4.2,.1)],fps=25)
    assert not cuts(timeline)


def test_secondary_audio_disagreement_keeps():
    timeline,_=decide(5,[(0,1),(4,5)],[Word('a',0,1,.9),Word('b',4,5,.9)],secondary_speech=[(1,4)])
    assert not cuts(timeline)


def test_unaligned_segment_keeps():
    timeline,_=decide(5,[(0,1),(4,5)],[Word('a',0,1,.9)],unaligned_ranges=[(1,5)])
    assert not cuts(timeline)


def test_no_asr_is_not_permission_to_delete():
    timeline,_=decide(5,[],[])
    assert not cuts(timeline)


def test_fast_mode_can_cut_activity_gap_but_cannot_delete_all():
    timeline,_=decide(5,[(0,1),(4,5)],[],mode='fast')
    assert cuts(timeline)
    timeline,_=decide(5,[],[],mode='fast')
    assert not cuts(timeline)


def test_high_confidence_isolated_filler_overrides_activity():
    words=[Word('Today',0,.7,.99),Word('um',1,1.4,.96),Word('topic',1.8,3,.99)]
    timeline,decisions=decide(3,[(0,3)],words,detect_fillers(words,'en'),fps=25)
    assert cuts(timeline)==[(.96,1.44)]
    assert decisions[0]['evidence']['audio_corroborated']


def test_uncertain_filler_and_emphasis_stay():
    words=[Word('Today',0,.7,.99),Word('hmm',1,1.4,.96),Word('topic',1.8,3,.99)]
    timeline,_=decide(3,[(0,3)],words,detect_fillers(words,'en'),fps=25)
    assert not cuts(timeline)
    words=[Word('very',0,.5,.99),Word('very',.8,1.3,.99),Word('good',1.5,3,.99)]
    timeline,_=decide(3,[(0,3)],words,detect_repetitions(words),fps=25)
    assert not cuts(timeline)


def test_exact_phrase_restart_removes_whole_earlier_phrase():
    words=[Word('I',0,.3,.98),Word('think',.31,.8,.98),Word('I',1.2,1.5,.98),
           Word('think',1.51,2,.98),Word('that',2.1,3,.98)]
    timeline,decisions=decide(3,[(0,3)],words,detect_repetitions(words,'conservative'),fps=25)
    assert cuts(timeline)==[(0,.84)]


def test_neighbor_word_margin_vetoes_lexical_cut():
    words=[Word('Today',0,.7,.99),Word('um',.75,1.15,.96),Word('topic',1.18,3,.99)]
    from editron_cut_engine.models import CutInterval
    candidate=CutInterval(.75,1.15,'filler_word',.99,word_indices=[1])
    timeline,decisions=decide(3,[(0,3)],words,[candidate],fps=25)
    assert not cuts(timeline)
    assert decisions[0]['decision_reason']=='protected_or_neighboring_word'


@pytest.mark.parametrize('text',['తెలుగు','हिन्दी','ಕನ್ನಡ','தமிழ்','മലയാളം'])
def test_multilingual_word_protection(text):
    timeline,_=decide(5,[(0,.2),(4.8,5)],[Word(text,.3,4.7,.7)],fps=25)
    assert not cuts(timeline)


@pytest.mark.parametrize('start,end',[(float('nan'),1),(0,float('inf')),(-1,2),(2,1),(0,6)])
def test_invalid_word_timestamps_fail_closed(start,end):
    with pytest.raises(ValueError):
        decide(5,[(0,5)],[Word('word',start,end,.9)])


def test_random_silence_proposals_never_cross_retained_words():
    rng=random.Random(20929)
    for _ in range(100):
        words=[Word(str(i),s,s+rng.uniform(.1,.5),.2) for i,s in enumerate(sorted(rng.uniform(.1,19) for _ in range(15)))]
        timeline,_=decide(20,[(0,.1),(19.9,20)],words,fps=25)
        assert validate_timeline(20,timeline)>0
        for s,e in cuts(timeline):
            assert all(min(e,w.end)-max(s,w.start)<=1e-7 for w in words)
            assert abs(s*25-round(s*25))<1e-6
            assert abs(e*25-round(e*25))<1e-6


@pytest.mark.parametrize('timeline',[
    [],[{'start':0,'end':2,'action':'SPEED','reason':'x'}],
    [{'start':0,'end':1,'action':'KEEP','reason':'x'},{'start':1.2,'end':2,'action':'KEEP','reason':'x'}],
    [{'start':0,'end':2,'action':'CUT','reason':'x'}],
    [{'start':0,'end':2,'action':'KEEP','reason':'x','output_start':1}],
    [{'start':0,'end':2,'action':'KEEP','reason':'x','confidence':float('nan')}],
])
def test_malformed_timeline_rejected(timeline):
    with pytest.raises(ValueError):
        validate_timeline(2,timeline)


def test_empty_interval_merge_regression():
    assert merge_intervals([])==[]
    assert merge_intervals([(1,1),(2,1)])==[]
    assert complement(4,[(0,1),(3,4)])==[(1.,3.)]
