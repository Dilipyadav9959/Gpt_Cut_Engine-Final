"""Exercise CLI dispatch and artifact safety without invoking model downloads."""

import json
import os
from unittest.mock import MagicMock

import pytest

from editron_cut_engine import cli


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"boundary test")
    plan = {"duration": 10, "cut_ranges": [], "repetition_cuts": [], "detected_language": "te"}
    engine = MagicMock()
    engine.analyze.return_value = plan
    engine.process.return_value = plan
    engine.render_plan.return_value = plan
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(cli, "CutEngine", factory)
    return source, engine, factory, plan


def test_analyze_only_writes_plan_without_rendering(invocation, capsys):
    source, engine, factory, plan = invocation
    assert cli.main([str(source), "--analyze-only", "--language", "te"]) == 0
    engine.analyze.assert_called_once_with(source, "te", protected_ranges=None, transcript_path=None)
    engine.process.assert_not_called()
    engine.render_plan.assert_not_called()
    destination = source.with_name("source.editron.analysis.json")
    assert json.loads(destination.read_text(encoding="utf-8")) == plan
    assert json.loads(capsys.readouterr().out)["output"] is None
    assert factory.call_args.kwargs["mode"] == "smart"


def test_legacy_invocation_and_preview_still_render(invocation):
    source, engine, _, _ = invocation
    output = source.parent / "result.mp4"
    preview = source.parent / "preview.json"
    assert cli.main(["--input", str(source), "--output", str(output), "--preview-json", str(preview)]) == 0
    assert engine.process.call_args.args == (source, output, "auto", preview)
    engine.analyze.assert_not_called()


def test_render_plan_does_not_analyze(invocation):
    source, engine, _, plan = invocation
    saved = source.parent / "saved.json"
    saved.write_text(json.dumps(plan), encoding="utf-8")
    cli.main([str(source), "--render-plan", str(saved), "--overwrite", "--fps", "30"])
    engine.analyze.assert_not_called()
    engine.process.assert_not_called()
    assert engine.render_plan.call_args.args[2] == plan
    assert engine.render_plan.call_args.kwargs["overwrite"] is True
    assert engine.render_plan.call_args.kwargs["fps"] == 30


def test_transcript_and_protected_ranges_passed_to_analysis(invocation):
    source, engine, _, _ = invocation
    protected = source.parent / "protected.json"
    ranges = [{"start": 2, "end": 3, "reason": "emphasis"}]
    protected.write_text(json.dumps(ranges), encoding="utf-8")
    transcript = source.parent / "transcript.json"
    cli.main([str(source), "--analyze-only", "--protect-json", str(protected),
              "--transcript-json", str(transcript)])
    assert engine.analyze.call_args.kwargs == {"protected_ranges": ranges, "transcript_path": transcript}


@pytest.mark.parametrize("arguments", [
    ["--output", "SOURCE"], ["--analyze-only", "--preview-json", "SOURCE"],
    ["--width", "1080"], ["--input", "SOURCE"], ["--mode", "pro"],
])
def test_invalid_or_destructive_arguments_rejected(invocation, arguments):
    source, engine, _, _ = invocation
    args = [str(source)] + [str(source) if arg == "SOURCE" else arg for arg in arguments]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    engine.process.assert_not_called()


def test_executable_overrides_are_scoped(invocation, monkeypatch):
    source, engine, _, plan = invocation
    monkeypatch.setenv("EDITRON_FFMPEG", "existing-ffmpeg")
    monkeypatch.delenv("EDITRON_FFPROBE", raising=False)

    def analyze(*args, **kwargs):
        assert os.environ["EDITRON_FFMPEG"] == "custom-ffmpeg"
        assert os.environ["EDITRON_FFPROBE"] == "custom-ffprobe"
        return plan

    engine.analyze.side_effect = analyze
    cli.main([str(source), "--analyze-only", "--ffmpeg", "custom-ffmpeg", "--ffprobe", "custom-ffprobe"])
    assert os.environ["EDITRON_FFMPEG"] == "existing-ffmpeg"
    assert "EDITRON_FFPROBE" not in os.environ


def test_engine_failure_is_reported(invocation, capsys):
    source, engine, _, _ = invocation
    engine.analyze.side_effect = RuntimeError("FFmpeg unavailable")
    with pytest.raises(SystemExit) as error:
        cli.main([str(source), "--analyze-only"])
    assert error.value.code == 1
    assert "FFmpeg unavailable" in capsys.readouterr().err
