"""Fixed public-validation diagnostic; not Samsung's official evaluation."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from evaluate import metrics
from localization import localize
from media import frames, probe
from model import ROOT, Scorer
from temporal_model import TemporalRecognizer


def seconds(value):
    h, m, s = map(float, value.split(":"))
    return 3600 * h + 60 * m + s


def main():
    root = ROOT / "work/temporal"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "work/recognition")
    parser.add_argument("--videos", type=Path, default=root)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/temporal/tsm_rgb.ckpt")
    args = parser.parse_args()
    data = args.data
    root.mkdir(parents=True, exist_ok=True)
    def read(name):
        with (data / name).open(encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    verbs = {int(r["id"]): r["key"] for r in read("EPIC_100_verb_classes.csv")}
    nouns = {int(r["id"]): r["key"] for r in read("EPIC_100_noun_classes.csv")}
    annotations = read("EPIC_100_validation.csv")
    training = {r["video_id"] for r in read("EPIC_100_train.csv")}
    temporal = TemporalRecognizer(args.checkpoint)
    clip = Scorer(ROOT / "code/vocabulary.json", 4)
    results = []
    for video_id in ("P28_18", "P09_08", "P07_17"):
        assert video_id not in training
        video = args.videos / (video_id + ".mp4")
        duration, _ = probe(video)
        truth = {"actions": [dict(verb=verbs[int(r["verb_class"])],
                                  noun=nouns[int(r["noun_class"])],
                                  start=seconds(r["start_timestamp"]),
                                  end=seconds(r["stop_timestamp"]))
                             for r in annotations if r["video_id"] == video_id]}
        started = time.perf_counter()
        actions = temporal.recognize(video, duration, args.checkpoint.parent / "classes.json")
        temporal_seconds = time.perf_counter() - started
        measured = metrics(actions, [], None, truth, duration)
        results.append(dict(video_id=video_id, model="tsm", seconds=temporal_seconds,
                            predictions=actions, metrics=measured, ground_truth_count=len(truth["actions"])))
        print(video_id, "TSM", temporal_seconds, measured, flush=True)
        started = time.perf_counter()
        scores, idle, embeddings, label = clip.predict(frames(video, 1))
        actions = localize(scores, idle, embeddings, clip.actions, duration, top_k=1, smoothing=True)
        results.append(dict(video_id=video_id, model="clip", seconds=time.perf_counter() - started,
                            predictions=actions, metrics=metrics(actions, [], label, truth, duration),
                            ground_truth_count=len(truth["actions"])))
        report = dict(scope="Public EPIC validation diagnostic; three fixed recordings; no training or tuning; not official scoring",
                      context_seconds=4, stride_seconds=1, sample_step_seconds=.5,
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      results=results)
        (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for name in ("tsm", "clip"):
        rows = [r for r in results if r["model"] == name]
        print(name, "mean macro F1", np.mean([r["metrics"][f"action_f1_tiou_{t}"] for r in rows for t in (.1, .3, .5)]), flush=True)


if __name__ == "__main__":
    main()
