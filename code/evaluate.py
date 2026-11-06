"""Local diagnostic metrics. These definitions are provisional until Samsung supplies scoring."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from media import frames, probe
from model import ROOT, Scorer
from localization import localize
from selection import FRAME_SECONDS, analyze
from selection_baseline import analyze as baseline


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def tiou(a, b):
    common = overlap(a, b)
    return common / max(a["end"] - a["start"] + b["end"] - b["start"] - common, 1e-9)


def metrics(actions, highlights, label, truth, duration):
    gt = truth["actions"]
    pairs = {(x["verb"], x["noun"]) for x in gt}
    captured = {(x["verb"], x["noun"]) for x in gt
                if sum(overlap(x, h) for h in highlights) >= .5 * (x["end"] - x["start"])}
    result = {"coverage_at_half_action": len(captured) / len(pairs) if pairs else None}
    for threshold in (.1, .3, .5):
        available = set(range(len(gt)))
        matches = 0
        for action in sorted(actions, key=lambda x: x["start"]):
            candidates = [i for i in available
                          if (gt[i]["verb"], gt[i]["noun"]) == (action["verb"], action["noun"])]
            best = max(candidates, key=lambda i: tiou(action, gt[i]), default=None)
            if best is not None and tiou(action, gt[best]) >= threshold:
                matches += 1
                available.remove(best)
        result[f"action_precision_tiou_{threshold}"] = matches / len(actions) if actions else 0
        result[f"action_recall_tiou_{threshold}"] = matches / len(gt) if gt else None
        result[f"action_f1_tiou_{threshold}"] = 2 * matches / (len(actions) + len(gt)) if actions or gt else None
    ticks = np.arange(.05, duration, .1)
    predicted = np.zeros(len(ticks), dtype=bool)
    expected = predicted.copy()
    for intervals, mask in ((highlights, predicted), (truth.get("highlights") or [], expected)):
        for item in intervals:
            mask |= (ticks >= item["start"]) & (ticks < item["end"])
    denominator = int(predicted.sum() + expected.sum())
    result["highlight_f1_at_10hz"] = 2 * int((predicted & expected).sum()) / denominator if denominator else 1
    if truth.get("highlights") is None:
        result["highlight_f1_at_10hz"] = None
    result["label_correct"] = label == truth["label"] if truth.get("label") is not None else None
    result["selected_fraction"] = sum(x["end"] - x["start"] for x in highlights) / duration
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "code/vocabulary.json")
    parser.add_argument("--top-k", type=int, choices=(1, 3), default=1)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-strength", type=float, default=0.0)
    args = parser.parse_args()
    scorer = Scorer(args.vocabulary, 4, args.adapter, args.adapter_strength)
    report = []
    for annotation in sorted(args.annotations.glob("*.json")):
        truth = json.loads(annotation.read_text(encoding="utf-8"))
        if truth["split"] != args.split:
            continue
        video = args.videos / (annotation.stem + ".mp4")
        duration, _ = probe(video)
        for field in ("actions", "highlights"):
            for row in truth.get(field) or []:
                if not 0 <= row["start"] < row["end"] <= duration:
                    raise ValueError(f"Invalid annotation interval in {annotation.name}")
        started = time.perf_counter()
        scores, idle, embeddings, label = scorer.predict(frames(video, FRAME_SECONDS))
        inference_seconds = time.perf_counter() - started
        for mode in ("baseline", "smoothed", "unsmoothed", "uniform"):
            started = time.perf_counter()
            if mode == "baseline":
                actions, highlights = baseline(scores, idle, embeddings, scorer.actions, duration)
            else:
                actions, highlights = analyze(scores, idle, embeddings, scorer.actions, duration,
                                              smoothing=mode != "unsmoothed")
            if mode == "uniform":
                width = duration * .15 / 10
                highlights = [{"start": (i + .5) * duration / 10 - width / 2,
                               "end": (i + .5) * duration / 10 + width / 2,
                               "importance": 0.0} for i in range(10)]
            if args.top_k > 1:
                actions = localize(scores, idle, embeddings, scorer.actions, duration,
                                   top_k=args.top_k, smoothing=mode != "unsmoothed")
            report.append(dict(video_id=video.stem, mode=mode, split=args.split,
                               action_count=len(truth["actions"]), predicted_count=len(actions),
                               vocabulary_pair_coverage=sum((x["verb"], x["noun"]) in
                                  {(a["verb"], a["noun"]) for a in scorer.actions}
                                  for x in truth["actions"]) / max(1, len(truth["actions"])),
                               feature_seconds_including_cache_lookup=inference_seconds,
                               selection_seconds=time.perf_counter() - started,
                               metrics=metrics(actions, highlights, label, truth, duration)))
    if not report:
        parser.error("No independent annotations found for the selected split.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
