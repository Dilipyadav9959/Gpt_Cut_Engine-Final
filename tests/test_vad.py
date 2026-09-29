import wave
import numpy as np
import pytest
from editron_cut_engine.vad import rms_speech


def wav(tmp_path,parts,channels=1):
    sr=16000
    arrays=[]
    for seconds,amplitude in parts:
        t=np.arange(round(sr*seconds))/sr
        arrays.append((np.sin(2*np.pi*440*t)*amplitude*32767).astype(np.int16))
    samples=np.concatenate(arrays)
    if channels==2:
        samples=np.column_stack([samples,-samples])
    path=tmp_path/'audio.wav'
    with wave.open(str(path),'wb') as f:
        f.setnchannels(channels); f.setsampwidth(2); f.setframerate(sr); f.writeframes(samples.tobytes())
    return path


def test_continuous_activity(tmp_path):
    ranges=rms_speech(wav(tmp_path,[(2,.4)]))
    assert ranges==[(0,2)]


def test_silence_and_short_gap(tmp_path):
    ranges=rms_speech(wav(tmp_path,[(1,.4),(.12,0),(1,.4),(1.5,0),(1,.4)]))
    assert len(ranges)==2
    assert ranges[0][1]==pytest.approx(2.13,abs=.03)
    assert ranges[1][0]==pytest.approx(3.62,abs=.03)


def test_leading_silence_not_filled(tmp_path):
    ranges=rms_speech(wav(tmp_path,[(.24,0),(1,.4)]))
    assert ranges[0][0]==pytest.approx(.24)


def test_silence_is_empty_activity(tmp_path):
    assert rms_speech(wav(tmp_path,[(1,0)]))==[]


def test_stereo_antiphase_keeps_activity(tmp_path):
    assert rms_speech(wav(tmp_path,[(1,.4)],2))==[(0,1)]
