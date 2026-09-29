"""Command line boundary for analysis, inspectable plans, and rendering."""

import argparse
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from . import __version__
from .engine import CutEngine
from .media import write_json


def build_parser():
    parser = argparse.ArgumentParser(
        prog="editron", description="Build and render conservative, explainable video edits."
    )
    parser.add_argument("source", nargs="?", type=Path, help="Source video")
    parser.add_argument("--input", type=Path, help="Legacy alias for the source video")
    parser.add_argument("--output", type=Path, help="Defaults to INPUT.editron.mp4")
    parser.add_argument("--version", action="version", version=f"EDITRON {__version__}")
    parser.add_argument("--mode", choices=["fast", "smart"], default="smart")
    parser.add_argument("--language", default="auto", help="Auto-detect or a Whisper language code, e.g. te, hi, en")
    parser.add_argument("--silence-mode", choices=["auto", "auto-editor", "vad", "rms", "off"], default="auto")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--rms-threshold-db", type=float, default=-38)
    parser.add_argument("--min-speech", type=float, default=0.15)
    parser.add_argument("--min-silence", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--min-pause", type=float, default=0.8, help="Minimum pause duration considered for removal")
    parser.add_argument("--min-cut", type=float, default=0.2)
    parser.add_argument("--repetitions", choices=["off", "conservative", "balanced", "aggressive"], default="conservative")
    parser.add_argument("--max-repeat-gap", type=float, default=1.8)
    parser.add_argument("--max-phrase-words", type=int, default=6)
    parser.add_argument("--no-fillers", action="store_true", help="Disable filler proposals")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--auto-editor-path", type=Path)
    parser.add_argument("--auto-editor-threshold", type=float, default=0.04)
    parser.add_argument("--auto-editor-margin", default="0.2s")
    parser.add_argument("--compare-baseline", action="store_true", help="Include an Auto-Editor comparison in the plan")
    parser.add_argument("--transcript-json", type=Path, help="Reuse word timestamps from a previous analysis/transcript JSON")
    parser.add_argument("--protect-json", type=Path, help="JSON list of {start, end, reason} ranges to retain, e.g. emphasis")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--analyze-only", action="store_true", help="Write an inspectable plan without rendering")
    operation.add_argument("--render-plan", type=Path, help="Validate and render a saved plan without transcription")
    parser.add_argument("--preview-json", type=Path, help="Write analysis JSON; also renders unless --analyze-only is supplied")
    parser.add_argument("--ffmpeg", type=Path, help="FFmpeg executable (or set EDITRON_FFMPEG)")
    parser.add_argument("--ffprobe", type=Path, help="FFprobe executable (or set EDITRON_FFPROBE)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of an existing rendered output")
    parser.add_argument("--width", type=int, help="Explicit output width; pair with --height")
    parser.add_argument("--height", type=int, help="Explicit output height; pair with --width")
    parser.add_argument("--fps", type=float, help="Explicit output frame rate; default preserves source timing")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], default="medium")
    parser.add_argument("--verbose", action="store_true")
    return parser


@contextmanager
def _media_environment(ffmpeg=None, ffprobe=None):
    overrides = {key: str(value) for key, value in (
        ("EDITRON_FFMPEG", ffmpeg), ("EDITRON_FFPROBE", ffprobe)
    ) if value is not None}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source and args.input:
        parser.error("Supply a positional source or --input, not both")
    source = args.source or args.input
    if source is None:
        parser.error("A source video is required")
    if not source.is_file():
        parser.error(f"Source video does not exist: {source}")
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")
    if args.render_plan and (args.transcript_json or args.protect_json):
        parser.error("--render-plan uses its saved decisions; omit --transcript-json and --protect-json")
    output = args.output or source.with_name(f"{source.stem}.editron.mp4")
    analysis_path = args.preview_json
    if args.analyze_only and analysis_path is None:
        analysis_path = source.with_name(f"{source.stem}.editron.analysis.json")
    destinations = [path for path in (output, analysis_path) if path is not None]
    inputs = [path for path in (source, args.render_plan, args.transcript_json, args.protect_json) if path is not None]
    if any(dest.resolve() == src.resolve() for dest in destinations for src in inputs):
        parser.error("Output and analysis paths must be different from all input files")
    if analysis_path and analysis_path.resolve() == output.resolve():
        parser.error("Output video and analysis JSON must have different paths")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        protected = _read_json(args.protect_json) if args.protect_json else None
        if protected is not None and not isinstance(protected, list):
            raise ValueError("--protect-json must contain a JSON list of ranges")
        with _media_environment(args.ffmpeg, args.ffprobe):
            engine = CutEngine(
                mode=args.mode, silence_mode=args.silence_mode,
                vad_threshold=args.vad_threshold, rms_threshold_db=args.rms_threshold_db,
                min_speech=args.min_speech, min_silence=args.min_silence, padding=args.padding,
                min_pause=args.min_pause, min_cut=args.min_cut,
                repetitions=args.repetitions, max_repeat_gap=args.max_repeat_gap,
                max_phrase_words=args.max_phrase_words, fillers=not args.no_fillers,
                whisper_model=args.whisper_model, device=args.device, compute_type=args.compute_type,
                auto_editor_path=str(args.auto_editor_path) if args.auto_editor_path else None,
                auto_editor_threshold=args.auto_editor_threshold,
                auto_editor_margin=args.auto_editor_margin, compare_baseline=args.compare_baseline,
            )
            render_options = dict(overwrite=args.overwrite, width=args.width, height=args.height,
                                  fps=args.fps, crf=args.crf, preset=args.preset)
            if args.render_plan:
                plan = engine.render_plan(source, output, _read_json(args.render_plan), **render_options)
                if analysis_path:
                    write_json(analysis_path, plan)
            elif args.analyze_only:
                plan = engine.analyze(source, args.language, protected_ranges=protected,
                                      transcript_path=args.transcript_json)
                write_json(analysis_path, plan)
            else:
                plan = engine.process(source, output, args.language, analysis_path,
                                      protected_ranges=protected, transcript_path=args.transcript_json,
                                      render_options=render_options)
        summary = {"output": None if args.analyze_only else str(output),
                   "analysis": str(analysis_path) if analysis_path else None,
                   "duration": plan.get("duration"), "cuts": len(plan.get("cut_ranges", [])),
                   "repetition_cuts": len(plan.get("repetition_cuts", [])),
                   "language": plan.get("detected_language"), "metrics": plan.get("metrics", {})}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        if args.verbose:
            logging.exception("Editron failed")
        parser.exit(1, f"editron: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
