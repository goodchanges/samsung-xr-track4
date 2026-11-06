import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

from demo import build_demo
from media import frames, probe, render
from model import ROOT, Scorer, load_supported_classes
from localization import localize
from output import clean_intervals
from selection import FRAME_SECONDS, analyze
from selection_baseline import analyze as baseline


def main():
    parser = argparse.ArgumentParser(description="Extract action highlights from a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "code" / "vocabulary.json")
    parser.add_argument("--supported-classes", type=Path, default=ROOT / "supported_classes.json",
                        help="Student-kit class JSON. Class mapping requires the kit format.")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--selector", choices=("preserved", "baseline", "duration", "temporal"), default="preserved")
    parser.add_argument("--unsmoothed", action="store_true")
    parser.add_argument("--top-k", type=int, choices=(1, 3), default=1)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-strength", type=float, default=0.0,
                        help="Visual adapter strength. Set a positive value to enable it (default: 0).")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--temporal-model", type=Path,
                        help="Optional EPIC TSM checkpoint; classes.json must be alongside it.")
    parser.add_argument("--temporal-context", type=int, choices=(2, 4),
                        help="Defaults to 2 for the temporal selector, otherwise 4.")
    parser.add_argument("--demo", type=Path, metavar="DIR")
    args = parser.parse_args()
    if args.selector == "temporal" and not args.temporal_model:
        parser.error("--selector temporal requires --temporal-model.")
    context = args.temporal_context or (2 if args.selector == "temporal" else 4)
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
        supported = load_supported_classes(args.supported_classes)
        if supported is not None:
            raise ValueError(f"Loaded {supported[0]}. Class mapping requires the confirmed student-kit format.")
        duration, audio = probe(source)
        if not 1 <= duration <= 600.05:
            raise ValueError("This prototype supports MP4 videos from 1 second to 10 minutes.")
        adapter = args.adapter
        if adapter is None and args.adapter_strength > 0 and not args.no_adapter and args.vocabulary.resolve() == (ROOT / "code/vocabulary.json").resolve():
            adapter = ROOT / "weights/action_adapter.npz"
        if args.no_adapter:
            adapter = None
        scorer = Scorer(args.vocabulary, args.threads, adapter, args.adapter_strength)
        raw = frames(source, FRAME_SECONDS)
        scores, idle, embeddings, label = scorer.predict(raw)
        actions, highlights = analyze(scores, idle, embeddings, scorer.actions, duration, smoothing=not args.unsmoothed)
        if args.selector == "baseline":
            _, highlights = baseline(scores, idle, embeddings, scorer.actions, duration)
        elif args.selector == "preserved":
            original = Scorer(ROOT / "code/vocabulary-baseline.json", args.threads, shared=scorer)
            old_scores, old_idle, _, _ = original.predict(raw)
            _, highlights = baseline(old_scores, old_idle, embeddings, original.actions, duration)
        if args.top_k > 1:
            actions = localize(scores, idle, embeddings, scorer.actions, duration,
                               top_k=args.top_k, smoothing=not args.unsmoothed)
        kitchen, margin, fraction = scorer.kitchen_confidence(embeddings)
        if args.temporal_model:
            print(f"Kitchen gate: accepted={kitchen} margin={margin:.4f} frame_fraction={fraction:.3f}", file=sys.stderr)
        if args.temporal_model and not kitchen:
            from temporal_selection import generic_select
            highlights = generic_select(actions, scores, scorer.actions, embeddings, duration, highlights)
            print("Route: generic CLIP + temporal selector", file=sys.stderr)
        if args.temporal_model and kitchen:
            print("Route: kitchen TSM", file=sys.stderr)
            try:
                from temporal_model import TemporalRecognizer
            except ImportError as exc:
                raise ValueError("Install optional model dependencies: python -m pip install -r requirements-temporal.txt") from exc
            temporal = TemporalRecognizer(args.temporal_model, args.threads)
            actions = temporal.recognize(source, duration, args.temporal_model.parent / "classes.json",
                                         context)
            if args.selector == "temporal":
                from temporal_selection import select
                verbs, nouns = temporal.score_video(source, duration, context)
                classes = json.loads((args.temporal_model.parent / "classes.json").read_text(encoding="utf-8"))
                highlights = select(actions, verbs, nouns, classes, embeddings, duration)
        actions, highlights = clean_intervals(actions, highlights, duration)
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
