"""Faster-Whisper adapter preserving multilingual evidence and GPU support."""

from functools import lru_cache
import logging
import math
from types import SimpleNamespace

from .models import Word

LOGGER = logging.getLogger(__name__)
COMPUTE_TYPES = {"auto", "default", "float32", "float16", "int8", "int8_float16",
                 "int8_float32", "int8_bfloat16", "bfloat16", "int16"}


def _cuda_available():
    # Faster-Whisper uses CTranslate2; PyTorch is not required for this backend.
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _load_model(model_name, device, compute_type, local_files_only=False):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device=device, compute_type=compute_type,
                        local_files_only=local_files_only)


def _cuda_runtime_failure(error):
    message = str(error).lower()
    return isinstance(error, (RuntimeError, OSError)) and any(
        token in message for token in (
            "cuda", "cublas", "cudnn", "nvidia", "libcud", "gpu out of memory",
        )
    )


def _validate_options(model_name, device, compute_type, language):
    if not str(model_name).strip():
        raise ValueError("whisper model must be nonempty")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if compute_type not in COMPUTE_TYPES:
        raise ValueError("Unsupported Whisper compute type")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise ValueError("language must be a language code or auto")


def _transcribe_once(audio_path, model_name, device, compute_type, language, local_files_only):
    model = _load_model(str(model_name), device, compute_type, local_files_only)
    iterator, info = model.transcribe(
        str(audio_path), language=None if language in (None, "auto") else language,
        word_timestamps=True, vad_filter=False, condition_on_previous_text=True,
    )
    words = []
    segments = []
    previous_start = -1.0
    for segment in iterator:
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("Whisper returned invalid segment timestamps")
        segments.append({
            "start": start, "end": end, "text": segment.text,
            "avg_logprob": getattr(segment, "avg_logprob", None),
            "no_speech_prob": getattr(segment, "no_speech_prob", None),
        })
        for item in segment.words or []:
            word = Word(item.word, float(item.start), float(item.end),
                        float(getattr(item, "probability", 0.0) or 0.0))
            if (not all(math.isfinite(value) for value in (word.start, word.end, word.probability))
                    or word.start < 0 or word.end < word.start
                    or word.start < previous_start or not 0 <= word.probability <= 1):
                raise ValueError("Whisper returned invalid word timestamps or confidence")
            # Zero-length/overlapping words are retained as uncertain evidence;
            # discarding them could authorize a destructive cut.
            previous_start = word.start
            words.append(word)
    return {
        "words": words, "segments": segments,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "transcript": "".join(segment["text"] for segment in segments).strip(),
        "device": device, "compute_type": compute_type,
        "model": str(model_name), "warnings": [],
    }


def transcribe_analysis(audio_path, model_name="large-v3", device="auto",
                        compute_type="auto", language=None, *, local_files_only=False):
    """Return segments, words, detected language and actual runtime metadata.

    Explicit CUDA requests fail visibly. Auto mode falls back only for GPU
    runtime/library errors, including errors during lazy segment decoding.
    """
    _validate_options(model_name, device, compute_type, language)
    selected_device = "cuda" if device == "auto" and _cuda_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    selected_compute = ("float16" if selected_device == "cuda" else "int8") if compute_type == "auto" else compute_type
    try:
        return _transcribe_once(audio_path, model_name, selected_device, selected_compute,
                                language, local_files_only)
    except (RuntimeError, OSError) as error:
        if device != "auto" or selected_device != "cuda" or not _cuda_runtime_failure(error):
            raise
        warning = f"Automatic GPU transcription failed; retried on CPU/int8: {error}"
        LOGGER.warning(warning)
        _load_model.cache_clear()
        analysis = _transcribe_once(audio_path, model_name, "cpu", "int8", language, local_files_only)
        analysis["warnings"].append(warning)
        return analysis


def transcribe_words(audio_path, model_name="large-v3", device="auto",
                     compute_type="auto", language=None):
    """Backward-compatible word-list and attribute-based language-info tuple."""
    analysis = transcribe_analysis(audio_path, model_name, device, compute_type, language)
    return analysis["words"], SimpleNamespace(
        language=analysis["language"], language_probability=analysis["language_probability"],
        device=analysis["device"], compute_type=analysis["compute_type"],
    )
