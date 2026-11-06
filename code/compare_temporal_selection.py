"""Compare local distinct-action coverage; no human highlight F1 is available."""
import argparse
import csv
import json

import numpy as np

from evaluate import metrics
from media import frames, probe
from model import ROOT, Scorer
from refine_temporal import DEVELOPMENT, EVALUATION
from selection_baseline import analyze
from temporal_experiment import seconds
from temporal_model import TemporalRecognizer, decode
from temporal_selection import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "evaluation"))
    args = parser.parse_args()
    root = ROOT / "work/temporal"
    config = json.loads((root / "selected-settings.json").read_text())["selected"]
    classes = json.loads((ROOT / "weights/temporal/classes.json").read_text())
    with (ROOT / "work/recognition/EPIC_100_validation.csv").open(encoding="utf-8-sig") as stream:
        annotations = list(csv.DictReader(stream))
    temporal = TemporalRecognizer(ROOT / "weights/temporal/tsm_rgb.ckpt")
    clip = Scorer(ROOT / "code/vocabulary-baseline.json", 4)
    results = []
    for video_id in DEVELOPMENT if args.split == "development" else EVALUATION:
        video = root / (video_id + ".mp4")
        duration, _ = probe(video)
        truth = {"actions": [dict(verb=classes["verb"][int(r["verb_class"])],
                                  noun=classes["noun"][int(r["noun_class"])],
                                  start=seconds(r["start_timestamp"]), end=seconds(r["stop_timestamp"]))
                             for r in annotations if r["video_id"] == video_id]}
        scores, idle, embeddings, label = clip.predict(frames(video, 1))
        _, preserved = analyze(scores, idle, embeddings, clip.actions, duration)
        verbs, nouns = temporal.score_video(video, duration, config["context"])
        actions = decode(verbs, nouns, classes, duration, config["smoothing"], config["top_k"])
        revised = select(actions, verbs, nouns, classes, embeddings, duration)
        budget = np.floor(duration * 1000 + 1e-6) * 145 // 1000 / 1000
        width = budget / 10
        uniform = [dict(start=float((i + .5) * duration / 10 - width / 2),
                        end=float((i + .5) * duration / 10 + width / 2), importance=0.) for i in range(10)]
        for name, highlights in (("preserved", preserved), ("temporal", revised), ("uniform", uniform)):
            measured = metrics(actions, highlights, label, truth, duration)
            results.append(dict(video_id=video_id, selector=name, config=config, highlights=highlights,
                                metrics=measured))
            print(video_id, name, "coverage", measured["coverage_at_half_action"], flush=True)
    (root / f"selection-{args.split}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name in ("preserved", "temporal", "uniform"):
        print(name, np.mean([r["metrics"]["coverage_at_half_action"] for r in results if r["selector"] == name]), flush=True)


if __name__ == "__main__":
    main()
