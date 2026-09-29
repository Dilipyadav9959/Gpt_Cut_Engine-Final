from types import SimpleNamespace
import unicodedata

import pytest

from editron_cut_engine import whisper
from editron_cut_engine.text import token_key


@pytest.mark.parametrize("left,right", [("కలం", "కాలం"), ("का", "की"), ("है", "हूँ")])
def test_indic_vowel_marks_distinguish_words(left, right):
    assert token_key(left) != token_key(right)


@pytest.mark.parametrize("text", ["తెలుగు", "हिन्दी", "ಕನ್ನಡ", "தமிழ்", "മലയാളം"])
def test_indic_normalization_preserves_all_letters_and_marks(text):
    assert token_key(text) == unicodedata.normalize("NFC", text)


def test_normalization_handles_canonical_equivalence_without_joining_words():
    assert token_key("CAFÉ!") == token_key("cafe\u0301")
    assert token_key("can-not") != token_key("cannot")


def result(device="cpu", compute_type="int8"):
    return {"words": [], "segments": [], "language": "te", "language_probability": .98,
            "transcript": "", "device": device, "compute_type": compute_type, "warnings": []}


def test_auto_cuda_detection_does_not_require_torch(monkeypatch):
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    assert whisper._cuda_available()


def test_auto_gpu_runtime_fallback_is_reported(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(whisper, "_cuda_available", lambda: True)
    def transcribe(audio, model, device, compute_type, language, local_files_only):
        calls.append((device, compute_type, language, local_files_only))
        if device == "cuda":
            raise RuntimeError("cuDNN library not found")
        return result(device, compute_type)
    monkeypatch.setattr(whisper, "_transcribe_once", transcribe)
    analysis = whisper.transcribe_analysis("clip.wav", language="te", local_files_only=True)
    assert calls == [("cuda", "float16", "te", True), ("cpu", "int8", "te", True)]
    assert analysis["warnings"] and "retried on CPU" in caplog.text


@pytest.mark.parametrize("device,message", [("cuda", "CUDA failed"), ("auto", "invalid audio input")])
def test_explicit_cuda_and_unrelated_errors_are_not_hidden(monkeypatch, device, message):
    monkeypatch.setattr(whisper, "_cuda_available", lambda: True)
    def fail(*args):
        raise RuntimeError(message)
    monkeypatch.setattr(whisper, "_transcribe_once", fail)
    with pytest.raises(RuntimeError, match=message):
        whisper.transcribe_analysis("clip.wav", device=device)


@pytest.mark.parametrize("options", [{"device": "magic"}, {"compute_type": "fake"},
                                      {"language": ""}, {"model_name": ""}])
def test_invalid_config_fails_before_loading_model(monkeypatch, options):
    def forbidden(*args):
        pytest.fail("must not load model for invalid options")
    monkeypatch.setattr(whisper, "_load_model", forbidden)
    with pytest.raises(ValueError):
        whisper.transcribe_analysis("clip.wav", **options)


def test_transcription_keeps_segment_confidence_and_indic_word_timing(monkeypatch):
    segment = SimpleNamespace(start=.1, end=1.5, text=" తెలుగు", avg_logprob=-.2, no_speech_prob=.02,
                              words=[SimpleNamespace(word=" తెలుగు", start=.1, end=1.5, probability=.96)])
    class Backend:
        def transcribe(self, audio_path, **options):
            assert options["language"] == "te"
            assert options["word_timestamps"]
            return iter([segment]), SimpleNamespace(language="te", language_probability=.98)
    monkeypatch.setattr(whisper, "_load_model", lambda *args: Backend())
    analysis = whisper.transcribe_analysis("clip.wav", device="cpu", language="te")
    assert analysis["transcript"] == "తెలుగు"
    assert analysis["words"][0].start == .1
    assert analysis["segments"][0]["no_speech_prob"] == .02
    assert analysis["device"] == "cpu"
    assert analysis["compute_type"] == "int8"


@pytest.mark.parametrize("start,end", [(float("nan"), .5), (.5, .1), (-.1, .5)])
def test_invalid_word_timestamps_raise(monkeypatch, start, end):
    segment = SimpleNamespace(start=0, end=1, text="word", words=[
        SimpleNamespace(word="word", start=start, end=end, probability=.95)])
    class Backend:
        def transcribe(self, *args, **kwargs):
            return iter([segment]), SimpleNamespace(language="en", language_probability=.98)
    monkeypatch.setattr(whisper, "_load_model", lambda *args: Backend())
    with pytest.raises(ValueError, match="invalid word"):
        whisper.transcribe_analysis("clip.wav", device="cpu")


def test_legacy_transcribe_words_preserves_tuple_contract(monkeypatch):
    monkeypatch.setattr(whisper, "transcribe_analysis", lambda *args: result())
    words, info = whisper.transcribe_words("clip.wav")
    assert words == [] and info.language == "te" and info.language_probability == .98
