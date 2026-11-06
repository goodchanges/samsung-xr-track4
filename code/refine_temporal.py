"""Select temporal settings on development recordings; evaluate once separately."""
import argparse
import csv
import json
import time
import hashlib
from datetime import datetime, timezone

import numpy as np

from evaluate import metrics
from localization import localize
from media import frames, probe
from model import ROOT, Scorer
from temporal_experiment import seconds
from temporal_model import TemporalRecognizer, decode

DEVELOPMENT = ["P28_18", "P09_08", "P07_17"]
EVALUATION = ["P18_10", "P08_14", "P04_28"]


def mean_f1(rows):
    return float(np.mean([r["metrics"][f"action_f1_tiou_{threshold}"]
                          for r in rows for threshold in (.1, .3, .5)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "evaluation"))
    args = parser.parse_args()
    root = ROOT / "work/temporal"
    classes = json.loads((ROOT / "weights/temporal/classes.json").read_text())
    with (ROOT / "work/recognition/EPIC_100_validation.csv").open(encoding="utf-8-sig") as stream:
        annotations = list(csv.DictReader(stream))
    with (ROOT / "work/recognition/EPIC_100_train.csv").open(encoding="utf-8-sig") as stream:
        train_ids = {r["video_id"] for r in csv.DictReader(stream)}
    settings_path = root / "selected-settings.json"
    if args.split == "development":
        configs = [dict(context=c, smoothing=s, top_k=k) for c in (4, 2) for s in (1, 3, 5) for k in (1, 2)]
        ids = DEVELOPMENT
    else:
        frozen_settings = settings_path.read_bytes()
        selected = json.loads(frozen_settings)["selected"]
        configs = [dict(context=4, smoothing=1, top_k=1)]
        if selected not in configs:
            configs.append(selected)
        ids = EVALUATION
    assert not set(DEVELOPMENT) & set(EVALUATION)
    assert not set(ids) & train_ids
    model = TemporalRecognizer(ROOT / "weights/temporal/tsm_rgb.ckpt")
    clip = Scorer(ROOT / "code/vocabulary.json", 4) if args.split == "evaluation" else None
    rows = []
    for video_id in ids:
        video = root / (video_id + ".mp4")
        duration, _ = probe(video)
        truth = {"actions": [dict(verb=classes["verb"][int(r["verb_class"])],
                                  noun=classes["noun"][int(r["noun_class"])],
                                  start=seconds(r["start_timestamp"]), end=seconds(r["stop_timestamp"]))
                              for r in annotations if r["video_id"] == video_id]}
        for context in sorted({c["context"] for c in configs}, reverse=True):
            started = time.perf_counter()
            verbs, nouns = model.score_video(video, duration, context)
            elapsed = time.perf_counter() - started
            for config in (c for c in configs if c["context"] == context):
                actions = decode(verbs, nouns, classes, duration, config["smoothing"], config["top_k"])
                rows.append(dict(video_id=video_id, config=config, model="tsm", duration=duration,
                                 inference_or_cache_seconds=elapsed, predictions=actions,
                                 ground_truth_count=len(truth["actions"]),
                                 metrics=metrics(actions, [], None, truth, duration)))
            print(video_id, context, "seconds context completed in", round(elapsed, 2), flush=True)
        if clip is not None:
            started = time.perf_counter()
            scores, idle, embeddings, label = clip.predict(frames(video, 1))
            actions = localize(scores, idle, embeddings, clip.actions, duration, top_k=1, smoothing=True)
            rows.append(dict(video_id=video_id, config=None, model="clip", duration=duration,
                             inference_or_cache_seconds=time.perf_counter() - started,
                             predictions=actions, ground_truth_count=len(truth["actions"]),
                             metrics=metrics(actions, [], label, truth, duration)))
        (root / f"refinement-{args.split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = [dict(config=c, mean_macro_action_f1=mean_f1([r for r in rows if r["config"] == c]))
               for c in configs]
    if args.split == "development":
        # Stable tie-break prefers baseline order, not ever more smoothing.
        best = max(summary, key=lambda item: item["mean_macro_action_f1"])
        settings_path.write_text(json.dumps(dict(selected=best["config"], development_ids=ids,
                                                 evaluation_ids=EVALUATION, summary=summary,
                                                 frozen_at_utc=datetime.now(timezone.utc).isoformat()), indent=2), encoding="utf-8")
    else:
        summary.append(dict(config="clip", mean_macro_action_f1=mean_f1([r for r in rows if r["model"] == "clip"])))
        (root / "evaluation-summary.json").write_text(json.dumps(dict(
            settings_sha256=hashlib.sha256(frozen_settings).hexdigest(),
            settings=json.loads(frozen_settings), summary=summary), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
