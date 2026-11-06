import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

from demo import build_demo
from media import frames, probe, render
from model import ROOT, Scorer
from selection import FRAME_SECONDS, analyze


def main():
    parser = argparse.ArgumentParser(description="Extract action highlights from a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "code" / "vocabulary.json")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--demo", type=Path, metavar="DIR")
    args = parser.parse_args()
    source, output = args.video.resolve(), args.output.resolve()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if output.suffix.lower() != ".json":
        parser.error("The output path must end in .json.")
    if not source.is_file() or source.suffix.lower() != ".mp4":
        parser.error("Supply an existing MP4 file.")
    if args.demo:
        folder = args.demo.resolve()
        reel = folder / "highlight.mp4"
        page = folder / "index.html"
        if reel == source:
            parser.error("The demo output would overwrite the input video.")
    started = perf_counter()
    try:
        duration, audio = probe(source)
        if not 1 <= duration <= 600.05:
            raise ValueError("This prototype supports MP4 videos from 1 second to 10 minutes.")
        scorer = Scorer(args.vocabulary, args.threads)
        raw = frames(source, FRAME_SECONDS)
        scores, idle, embeddings, label = scorer.predict(raw)
        actions, highlights = analyze(scores, idle, embeddings, scorer.actions, duration)
        result = dict(video_id=source.stem, label=label, actions=actions, highlights=highlights)
        if args.demo:
            render(source, reel, highlights, audio)
            build_demo(page, source, reel, result, duration)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Track 4: {exc}\n")
    selected = sum(item["end"] - item["start"] for item in highlights)
    print(f"{len(actions)} actions · {selected:.2f}/{duration:.2f}s selected "
          f"({selected/duration:.1%}) · {perf_counter()-started:.1f}s elapsed", file=sys.stderr)


if __name__ == "__main__":
    main()
