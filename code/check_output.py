"""Provisional local checker. The challenge validator is supplied separately."""
import argparse
import json
import math
from pathlib import Path

from media import probe


def check(document, video_id, duration):
    if set(document) != {"video_id", "label", "actions", "highlights"}:
        raise ValueError("Unexpected provisional top-level keys.")
    if document["video_id"] != video_id or not isinstance(document["label"], str):
        raise ValueError("Unexpected video ID or label.")
    total = 0
    previous_end = 0
    for name, extra in (("actions", ("verb", "noun")), ("highlights", ("importance",))):
        if not isinstance(document[name], list):
            raise ValueError(f"{name} must be an array.")
        for row in document[name]:
            if set(row) != {"start", "end", *extra}:
                raise ValueError(f"Unexpected keys in {name} interval.")
            start, end = row["start"], row["end"]
            if not all(type(n) in (int, float) and math.isfinite(n) for n in (start, end)):
                raise ValueError(f"Non-finite or non-numeric time in {name}.")
            if not 0 <= start < end <= duration:
                raise ValueError(f"Out of range interval in {name}.")
            if name == "actions":
                if not all(isinstance(row[key], str) and row[key] for key in extra):
                    raise ValueError("Missing verb or noun.")
            else:
                if start < previous_end:
                    raise ValueError("Highlights must be sorted and nonoverlapping.")
                importance = row["importance"]
                if type(importance) not in (int, float) or not 0 <= importance <= 1:
                    raise ValueError("Importance must be between zero and one.")
                previous_end = end
                total += end - start
    if total > duration * 0.15 + 1e-8:
        raise ValueError("Highlight duration exceeds 15%.")
    print(f"Provisional check passed ({total:.3f}/{duration:.3f}s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    duration, _ = probe(args.video)
    check(json.loads(args.output.read_text(encoding="utf-8")), args.video.stem, duration)
