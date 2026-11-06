"""Choose recognition settings on development participants; evaluate frozen choice once."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import metrics
from localization import localize
from media import frames, probe
from model import Scorer
from selection_baseline import analyze


def summarize(rows):
    keys = ["action_precision_tiou_0.1", "action_recall_tiou_0.1", "action_f1_tiou_0.1",
            "action_precision_tiou_0.3", "action_recall_tiou_0.3", "action_f1_tiou_0.3",
            "action_precision_tiou_0.5", "action_recall_tiou_0.5", "action_f1_tiou_0.5",
            "coverage_at_half_action"]
    return {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys}


def run(scorer, root, split, configs, adapter=None):
    result = {json.dumps(config, sort_keys=True): [] for config in configs}
    for path in sorted((root / "annotations").glob("*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        if truth["split"] != split:
            continue
        video = root / "videos" / (path.stem + ".mp4")
        duration, _ = probe(video)
        scores, idle, embeddings, label = scorer.predict(frames(video, 1))
        for config in configs:
            adjusted = scores.copy()
            if config["strength"]:
                similarities = embeddings @ adapter["prototypes"].T
                adjusted[:, adapter["indices"]] += config["strength"] * (similarities - similarities.mean(axis=1, keepdims=True))
            actions = localize(adjusted, idle, embeddings, scorer.actions, duration,
                               top_k=config["top_k"], smoothing=config["smoothing"])
            _, highlights = analyze(adjusted, idle, embeddings, scorer.actions, duration)
            result[json.dumps(config, sort_keys=True)].append(dict(
                video_id=path.stem, duration=duration, action_count=len(truth["actions"]),
                prediction_count=len(actions), metrics=metrics(actions, highlights, label, truth, duration)))
        print(f"Evaluated {split}: {path.stem}", flush=True)
    return [dict(config=json.loads(key), videos=rows, macro=summarize(rows)) for key, rows in result.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("expanded", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    base = dict(strength=0, top_k=1, smoothing=True)
    baseline = Scorer(args.baseline, 4)
    base_dev = run(baseline, args.root, "development", [base])[0]
    expanded = Scorer(args.expanded, 4)
    with np.load(args.adapter, allow_pickle=False) as saved:
        adapter = {key: saved[key] for key in ("indices", "prototypes")}
    configs = [dict(strength=strength, top_k=k, smoothing=smoothing)
               for strength in (0, .1, .25, .5) for k in (1, 3) for smoothing in (True, False)]
    candidates = run(expanded, args.root, "development", configs, adapter)
    def objective(row):
        return float(np.mean([row["macro"][f"action_f1_tiou_{t}"] for t in (.1, .3, .5)]))
    winner = max(candidates, key=objective)
    choice = dict(config=winner["config"], objective="Macro action F1 averaged over tIoU 0.1, 0.3, 0.5",
                  baseline_development=objective(base_dev), revised_development=objective(winner))
    # Persist selection BEFORE accessing held-out evaluation annotations.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".selection.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    baseline_eval = run(baseline, args.root, "evaluation", [base])[0]
    revised_eval = run(expanded, args.root, "evaluation", [winner["config"]], adapter)[0]
    report = dict(scope="Local participant-held-out experiment on EPIC training split; NOT official hackathon scoring",
                  selection=choice, baseline_development=base_dev, candidates=candidates,
                  baseline_evaluation=baseline_eval, revised_evaluation=revised_eval)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(selection=choice, baseline=baseline_eval["macro"], revised=revised_eval["macro"]), indent=2))


if __name__ == "__main__":
    main()
