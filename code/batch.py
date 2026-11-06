import argparse
import json
from pathlib import Path

from check_output import check
from media import frames, probe
from model import ROOT, Scorer
from selection import FRAME_SECONDS, analyze
from selection_baseline import analyze as baseline


def main():
    parser = argparse.ArgumentParser(description="Process a directory with one shared model.")
    parser.add_argument("videos", type=Path)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "code" / "vocabulary.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--selector", choices=("baseline", "duration"), default="baseline")
    parser.add_argument("--unsmoothed", action="store_true")
    args = parser.parse_args()
    if not args.videos.is_dir():
        parser.error("Supply an existing video directory.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    videos = sorted(p for p in args.videos.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    if not videos:
        parser.error("No MP4 videos found.")
    if len({p.stem.casefold() for p in videos}) != len(videos):
        parser.error("Duplicate video IDs.")
    scorer = Scorer(args.vocabulary, args.threads)
    args.outputs.mkdir(parents=True, exist_ok=True)
    for video in videos:
        duration, _ = probe(video)
        if not 1 <= duration <= 600.05:
            parser.error(f"Unsupported duration for {video.name}")
        scores, idle, embeddings, label = scorer.predict(frames(video, FRAME_SECONDS))
        actions, highlights = analyze(scores, idle, embeddings, scorer.actions, duration,
                                      smoothing=not args.unsmoothed)
        if args.selector == "baseline":
            _, highlights = baseline(scores, idle, embeddings, scorer.actions, duration)
        result = dict(video_id=video.stem, label=label, actions=actions, highlights=highlights)
        check(result, video.stem, duration)
        target = args.outputs / (video.stem + ".json")
        import os
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
