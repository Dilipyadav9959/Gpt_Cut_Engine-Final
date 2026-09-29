import pytest
from editron_cut_engine.auto_editor import parse_v1
from editron_cut_engine.evaluation import cut_metrics, compare
from editron_cut_engine.models import Word


def test_auto_editor_actual_316_format():
    raw={'version':'1','timebase':'25/1','chunks':[[0,25,1],[25,75,99999],[75,100,1]]}
    assert parse_v1(raw,4,25)==[(0,1),(3,4)]
    raw['chunks'][1][2]=0
    assert parse_v1(raw,4,25)==[(0,1),(3,4)]


@pytest.mark.parametrize('chunks',[[[0,25,2]],[[1,25,1]],[[0,25,1],[26,100,1]],[[0,1000,1]],[[0,float('nan'),1]]])
def test_bad_auto_editor_timelines_rejected(chunks):
    with pytest.raises(ValueError):
        parse_v1({'version':'1','timebase':'25/1','chunks':chunks},4,25)


def test_metrics_need_human_labels():
    m=cut_metrics([(1,3)],[Word('quiet',.5,1.5,.5)])
    assert m['word_clipping_proxy']==1
    assert m['false_cut_seconds'] is None
    assert m['naturalness'] is None


def test_duration_evaluation_with_reviewed_labels():
    m=cut_metrics([(1,3),(5,6)],reference=[(2,4)])
    assert m['false_cut_seconds']==2
    assert m['missed_cut_seconds']==1
    assert m['cut_precision']==pytest.approx(1/3)
    assert m['cut_recall']==.5
    assert m['unnecessary_cut_count']==1
