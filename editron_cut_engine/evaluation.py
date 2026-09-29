"""Duration-based evaluation and timestamp proxies; human scores remain explicit."""
import argparse
import json
from pathlib import Path
from .timeline import merge_intervals, validate_ranges


def interval_duration(ranges):
    return sum(b-a for a, b in merge_intervals(ranges))


def intersection_duration(a, b):
    return sum(max(0., min(y, v)-max(x, u)) for x, y in merge_intervals(a)
               for u, v in merge_intervals(b))


def cut_metrics(cuts, words=(), reference=None, naturalness=None):
    cuts = merge_intervals(cuts)
    total = interval_duration(cuts)
    # A boundary strictly inside a word is a clipping proxy. Full word deletions
    # are counted separately; legitimate filler edits can remove whole words.
    clipped = sum(any(w.start+1e-6 < b < w.end-1e-6 for s, e in cuts for b in (s, e)) for w in words)
    overlapped = sum(any(min(e, w.end)-max(s, w.start)>1e-6 for s, e in cuts) for w in words)
    result = {"cut_count":len(cuts), "removed_duration":total,
              "average_removed_duration": total/len(cuts) if cuts else 0.,
              "word_clipping_proxy":clipped, "words_overlapping_cuts":overlapped,
              "false_cut_seconds":None, "missed_cut_seconds":None,
              "unnecessary_cut_count":None, "cut_precision":None, "cut_recall":None,
              "naturalness":naturalness, "rendering_success":None,
              "synchronization_error_seconds":None}
    if reference is not None:
        true = intersection_duration(cuts, reference)
        target = interval_duration(reference)
        result.update(false_cut_seconds=max(0., total-true), missed_cut_seconds=max(0., target-true),
                      unnecessary_cut_count=sum(intersection_duration([(s,e)],reference)==0 for s,e in cuts),
                      cut_precision=true/total if total else (1. if not target else None),
                      cut_recall=true/target if target else None)
    return result


def compare(plan, reference=None, naturalness=None):
    from .models import Word
    words = [Word(**w) for w in plan.get("words", [])]
    edits = [(x["start"],x["end"]) for x in plan["cut_ranges"]]
    result = {"duration":plan["duration"], "editron":cut_metrics(edits,words,reference,naturalness),
              "auto_editor":None,
              "notes":["Word clipping is an ASR-timestamp proxy, not a phoneme-level guarantee.",
                       "False/missed cuts require human safe-cut labels; naturalness requires a human rating."]}
    if plan.get("baseline"):
        baseline = [(x["start"], x["end"]) for x in plan["baseline"]["cut_ranges"]]
        result["auto_editor"] = cut_metrics(baseline, words, reference)
    rendering = plan.get("rendering")
    if rendering:
        result["editron"]["rendering_success"] = True
        result["editron"]["synchronization_error_seconds"] = rendering.get("audio_video_end_delta")
    return result


def main():
    p=argparse.ArgumentParser(description="Compare Editron and Auto-Editor against reviewed safe-cut labels")
    p.add_argument("plan"); p.add_argument("--reference"); p.add_argument("--output")
    args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
    reference=naturalness=None
    if args.reference:
        labels=json.loads(Path(args.reference).read_text(encoding="utf-8-sig"))
        reference=[(x["start"],x["end"]) for x in labels["safe_cuts"]]
        validate_ranges(plan["duration"],reference)
        naturalness=labels.get("naturalness")
        if naturalness is not None and (not isinstance(naturalness,(int,float)) or not 1 <= naturalness <= 5):
            raise ValueError("Naturalness must be a human score from 1 to 5")
    result=compare(plan,reference,naturalness)
    if args.output:
        from .media import write_json
        write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
