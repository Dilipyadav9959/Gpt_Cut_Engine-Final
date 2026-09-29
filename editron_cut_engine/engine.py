"""Orchestration only: media -> evidence -> decisions -> validated plan -> render."""
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path

from .auto_editor import analyze_activity
from .decisions import decide, validate_words
from .evaluation import compare
from .fillers import detect_fillers
from .media import extract_wav, probe, render, write_json, resolve_binary
from .models import KeepInterval, Word
from .repetition import detect_repetitions
from .timeline import complement, validate_timeline
from .vad import rms_speech, silero_speech
from .whisper import transcribe_analysis

LOGGER = logging.getLogger(__name__)


def source_identity(path, media):
    p=Path(path).resolve()
    stat=p.stat()
    return {"path":str(p),"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,
            "duration":media["duration"]}


def load_transcript(path, source, duration):
    data=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data,dict) or "words" not in data:
        raise ValueError("Transcript JSON must contain words and source duration")
    if not isinstance(data.get("duration"),(int,float)) or abs(data["duration"]-duration)>1e-3:
        raise ValueError("Transcript duration does not match source media")
    if data.get("source"):
        for key in ("size_bytes","mtime_ns"):
            if data["source"].get(key)!=source[key]:
                raise ValueError("Transcript belongs to a different or modified media source")
    warnings=[] if data.get("source") else ["Imported transcript has no source fingerprint; source association is caller-supplied."]
    words=[Word(**{k:w[k] for k in ("text","start","end","probability") if k in w}) for w in data["words"]]
    validate_words(words,duration)
    return {"words":words,"segments":data.get("segments",[]),
            "language":data.get("language",data.get("detected_language")),
            "language_probability":data.get("language_probability"),
            "transcript":data.get("transcript"," ".join(w.text.strip() for w in words)),
            "device":"cached","compute_type":"cached","model":data.get("model","imported"),
            "warnings":warnings}


class CutEngine:
    def __init__(self, silence_mode="auto", vad_threshold=.5, rms_threshold_db=-38,
                 min_speech=.15, min_silence=.25, padding=.08,
                 repetitions="conservative", max_repeat_gap=1.8, max_phrase_words=6,
                 whisper_model="large-v3", device="auto", compute_type="auto",
                 mode="smart", min_pause=.8, min_cut=.2, fillers=True,
                 auto_editor_path=None, auto_editor_threshold=.04,
                 auto_editor_margin="0.2s", compare_baseline=False):
        self.cfg=locals().copy(); self.cfg.pop("self")
        for key,choices in {"silence_mode":{"auto","auto-editor","vad","rms","off"},
                            "mode":{"fast","smart"},"repetitions":{"off","conservative","balanced","aggressive"},
                            "device":{"auto","cpu","cuda"}}.items():
            if self.cfg[key] not in choices:
                raise ValueError(f"Invalid {key}: {self.cfg[key]}")
        for key in ("vad_threshold","rms_threshold_db","min_speech","min_silence","padding",
                    "max_repeat_gap","min_pause","min_cut","auto_editor_threshold"):
            value=self.cfg[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if key!="rms_threshold_db" and value<0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0<vad_threshold<1 or not 0<auto_editor_threshold<=1 or not -120<=rms_threshold_db<=0:
            raise ValueError("Invalid audio threshold")
        if min_cut<=0 or min_pause<min_cut or max_repeat_gap<=0:
            raise ValueError("Require 0 < min_cut <= min_pause and positive repeat gap")
        if isinstance(max_phrase_words,bool) or not isinstance(max_phrase_words,int) or not 1<=max_phrase_words<=20:
            raise ValueError("max_phrase_words must be an integer from 1 to 20")
        if mode=="fast" and repetitions!="off":
            LOGGER.info("Fast mode disables transcription and semantic edits")

    def analyze(self,input_path,language="auto",protected_ranges=None,transcript_path=None):
        started=time.perf_counter()
        input_path=Path(input_path).resolve()
        media=probe(input_path)
        dur=media["duration"]
        source=source_identity(input_path,media)
        warnings=[]
        detector={"backend":"off"}
        baseline=None
        asr={"words":[],"segments":[],"language":None,"language_probability":None,
             "transcript":"","device":None,"compute_type":None,"model":None,"warnings":[]}
        speech=[(0.,dur)]
        secondary=None
        c=self.cfg
        if media["has_audio"]:
            with tempfile.TemporaryDirectory(prefix="editron_analysis_") as td:
                wav=Path(td)/"audio.wav"
                extract_wav(input_path,wav)
                rms=rms_speech(wav,c["rms_threshold_db"],30,c["min_speech"],c["min_silence"])
                selected=c["silence_mode"]
                auto_binary=c["auto_editor_path"]
                if selected=="auto" or c["compare_baseline"]:
                    if not auto_binary:
                        try:
                            auto_binary=resolve_binary("auto-editor")
                        except FileNotFoundError:
                            if c["compare_baseline"]:
                                raise
                    if selected=="auto":
                        selected="auto-editor" if auto_binary else "vad"
                        if not auto_binary:
                            warnings.append("Auto-Editor unavailable; selected local VAD/RMS backend.")
                if selected=="auto-editor" or c["compare_baseline"]:
                    auto_ranges,auto_meta=analyze_activity(input_path,dur,media.get("fps") or 30,
                        executable=auto_binary,threshold=c["auto_editor_threshold"],margin=c["auto_editor_margin"])
                    baseline={**auto_meta,"keep_ranges":[{"start":s,"end":e} for s,e in auto_ranges],
                              "cut_ranges":[{"start":s,"end":e} for s,e in complement(dur,auto_ranges)]}
                if selected=="auto-editor":
                    speech=auto_ranges; detector={k:v for k,v in auto_meta.items() if k!="timeline"}
                    secondary=rms
                elif selected=="vad":
                    speech=silero_speech(wav,c["vad_threshold"],c["min_speech"],c["min_silence"])
                    if speech is None:
                        warnings.append("Silero VAD is not installed; RMS activity used explicitly as fallback.")
                        speech=rms; detector={"backend":"rms","threshold_db":c["rms_threshold_db"]}
                    else:
                        detector={"backend":"silero"}; secondary=rms
                elif selected=="rms":
                    speech=rms; detector={"backend":"rms","threshold_db":c["rms_threshold_db"]}
                if c["mode"]=="smart":
                    LOGGER.info("Transcribing %s (%s)",input_path.name,c["whisper_model"])
                    asr=load_transcript(transcript_path,source,dur) if transcript_path else transcribe_analysis(
                        wav,c["whisper_model"],c["device"],c["compute_type"],language)
                    warnings.extend(asr.get("warnings",[]))
                else:
                    warnings.append("Fast mode has no transcript protection; review audio-only cuts before publishing.")
        else:
            warnings.append("Source has no audio; all media retained.")
        words=asr["words"]
        validate_words(words,dur)
        unaligned=[]
        for seg in asr["segments"]:
            s,e=seg["start"],seg["end"]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (s,e)) or not 0<=s<=e<=dur+1e-6:
                raise ValueError("Transcript segment outside source media")
            if e>s and str(seg.get("text","")).strip() and not any(w.start<e and w.end>s for w in words):
                unaligned.append((s,e))
        if unaligned:
            warnings.append("Transcript segments without aligned words protected in full.")
        candidates=[]
        if c["mode"]=="smart":
            candidates.extend(detect_repetitions(words,c["repetitions"],c["max_repeat_gap"],c["max_phrase_words"]))
            if c["fillers"]:
                candidates.extend(detect_fillers(words,asr["language"]))
        timeline,decisions=decide(dur,speech,words,candidates,fps=media.get("fps") or 0,
            padding=c["padding"],min_pause=c["min_pause"],min_cut=c["min_cut"],
            protected_ranges=protected_ranges or [],unaligned_ranges=unaligned,
            secondary_speech=secondary,mode=c["mode"])
        kept=[x for x in timeline if x["action"]=="KEEP"]
        cuts=[x for x in timeline if x["action"]=="CUT"]
        final_duration=validate_timeline(dur,timeline)
        plan={"version":"2.0","engine":"editron","source":source,"media":media,"duration":dur,
              "detected_language":asr["language"],"language_probability":asr["language_probability"],
              "transcript":asr["transcript"],"segments":asr["segments"],"words":[w.to_dict() for w in words],
              "transcription":{k:asr[k] for k in ("model","device","compute_type")},
              "activity":detector,"speech_ranges":[{"start":s,"end":e} for s,e in speech],
              "silence_ranges":[{"start":s,"end":e} for s,e in complement(dur,speech)],
              "protected_ranges":protected_ranges or [],"decisions":decisions,"timeline":timeline,
              "keep_ranges":kept,"cut_ranges":cuts,
              "repetition_cuts":[x for x in cuts if x["reason"]=="repeated_word_or_phrase"],
              "baseline":baseline,"warnings":warnings,"config":dict(c),
              "summary":{"input_duration":dur,"output_duration":final_duration,
                         "removed_duration":dur-final_duration,"cut_count":len(cuts),
                         "proposed_edits":len(decisions),"accepted_edits":sum(x["status"]=="accepted" for x in decisions),
                         "rejected_edits":sum(x["status"]=="rejected" for x in decisions),
                         "analysis_seconds":time.perf_counter()-started}}
        plan["evaluation"]=compare(plan)
        for warning in warnings:
            LOGGER.warning(warning)
        LOGGER.info("Analysis: %.2fs -> %.2fs; %d accepted cuts",dur,final_duration,len(cuts))
        return plan

    def render_plan(self,input_path,output_path,plan,**render_options):
        media=probe(input_path)
        if plan.get("version")!="2.0":
            raise ValueError("Rendering saved plans requires a version 2.0 Editron timeline")
        identity=source_identity(input_path,media)
        for key in ("size_bytes","mtime_ns","duration"):
            if plan.get("source",{}).get(key)!=identity[key]:
                raise ValueError("Plan does not match the current source media")
        validate_timeline(media["duration"],plan["timeline"])
        keeps=[KeepInterval(x["start"],x["end"],x["reason"]) for x in plan["timeline"] if x["action"]=="KEEP"]
        # Renderer executes this timeline only; it never invokes an analyzer.
        plan["rendering"]=render(input_path,output_path,keeps,**render_options)
        plan["evaluation"]=compare(plan)
        return plan

    def process(self,input_path,output_path,language="auto",preview_json=None,
                protected_ranges=None,transcript_path=None,render_options=None):
        if preview_json and Path(preview_json).resolve() in (Path(input_path).resolve(),Path(output_path).resolve()):
            raise ValueError("Analysis JSON must differ from input and output media")
        plan=self.analyze(input_path,language,protected_ranges,transcript_path)
        if preview_json:
            write_json(preview_json,plan)
        try:
            self.render_plan(input_path,output_path,plan,**(render_options or {}))
        except Exception as error:
            plan["error"]={"stage":"render","type":type(error).__name__,"message":str(error)}
            if preview_json:
                write_json(preview_json,plan)
            raise
        if preview_json:
            write_json(preview_json,plan)
        return plan
