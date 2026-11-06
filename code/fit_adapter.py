"""Fit small visual prototypes using only annotations explicitly marked fit."""
import argparse
import json
from pathlib import Path

import numpy as np

from media import frames
from model import Scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("vocabulary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scorer = Scorer(args.vocabulary, 4)
    pairs = {(x["verb"], x["noun"]): i for i, x in enumerate(scorer.actions)}
    vectors = {}
    fitted = []
    for path in sorted(args.annotations.glob("*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        if truth["split"] != "fit":
            continue
        _, _, embeddings, _ = scorer.predict(frames(args.videos / (path.stem + ".mp4"), 1))
        times = np.arange(len(embeddings), dtype=float)
        for action in truth["actions"]:
            index = pairs.get((action["verb"], action["noun"]))
            selected = (times >= action["start"]) & (times < action["end"])
            if index is not None and selected.any():
                vectors.setdefault(index, []).append(embeddings[selected].mean(axis=0))
        fitted.append(path.stem)
        print(f"Fitted {path.stem}", flush=True)
    if not vectors:
        parser.error("No supported fit actions found.")
    indices = sorted(vectors)
    prototypes = np.stack([np.mean(vectors[i], axis=0) for i in indices])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, indices=np.array(indices), prototypes=prototypes,
                        vocabulary_signature=scorer.vocabulary_signature.decode(),
                        fit_video_ids=np.array(fitted))
    print(f"Saved {len(indices)} action prototypes from {len(fitted)} videos.")


if __name__ == "__main__":
    main()
