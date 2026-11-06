"""Development-only object evidence experiment using frozen CLIP weights."""
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import metrics
from localization import localize
from media import frames, probe
from model import ROOT, Scorer


def main():
    scorer = Scorer(ROOT / "code/vocabulary.json", 4)
    nouns = sorted({action["noun"] for action in scorer.actions})
    indices = [nouns.index(action["noun"]) for action in scorer.actions]
    prompts = [
        "A close-up photo of a " + " ".join(reversed(noun.split(":"))).replace("-", " ") + "."
        for noun in nouns
    ]
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(prompts), 32):
            tokens = scorer.processor(text=prompts[offset:offset + 32], padding=True,
                                      truncation=True, return_tensors="pt")
            vectors.append(scorer.model.get_text_features(**tokens))
        text = torch.nn.functional.normalize(torch.cat(vectors), dim=-1).numpy()
    results = []
    root = ROOT / "work/recognition"
    for video_id in json.loads((ROOT / "docs/recognition/splits.json").read_text())["development"]:
        truth = json.loads((root / "annotations" / (video_id + ".json")).read_text())
        video = root / "videos" / (video_id + ".mp4")
        duration, _ = probe(video)
        scores, idle, embeddings, label = scorer.predict(frames(video, 1))
        objects = embeddings @ text.T
        evidence = objects[:, indices] - objects.mean(axis=1, keepdims=True)
        for strength in (0.0, 0.1, 0.25, 0.5):
            actions = localize(scores + strength * evidence, idle, embeddings,
                               scorer.actions, duration, top_k=1, smoothing=True)
            measured = metrics(actions, [], label, truth, duration)
            results.append(dict(video_id=video_id, strength=strength, metrics=measured))
        print(f"Completed {video_id}", flush=True)
    summary = []
    for strength in (0.0, 0.1, 0.25, 0.5):
        rows = [row for row in results if row["strength"] == strength]
        score = float(np.mean([row["metrics"][f"action_f1_tiou_{t}"]
                               for row in rows for t in (.1, .3, .5)]))
        summary.append(dict(strength=strength, mean_macro_action_f1=score))
    report = dict(scope="Development experiment only; frozen weights; no highlight evaluation",
                  summary=summary, videos=results)
    target = root / "object-experiment.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
