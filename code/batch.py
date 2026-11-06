import argparse
import json
import csv
import subprocess
import sys
from pathlib import Path

from check_output import check
from media import frames, probe
from model import ROOT, Scorer, load_supported_classes
from localization import localize
from output import clean_intervals
from selection import FRAME_SECONDS, analyze
from selection_baseline import analyze as baseline


def manifest_entries(manifest, root, id_column="video_id", path_column="path"):
    """Use explicit manifest IDs, never guess dataset paths or silently skip rows."""
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {id_column, path_column}.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns {id_column!r}, {path_column!r}; set column flags to match the kit.")
        entries = []
        seen = set()
        for row in reader:
            video_id = row[id_column].strip()
            if not video_id or video_id in (".", "..") or any(c in video_id for c in '/\\:*?"<>|'):
                raise ValueError("Manifest contains an unsafe or empty video ID.")
            if video_id.casefold() in seen:
                raise ValueError(f"Duplicate manifest video ID: {video_id}")
            seen.add(video_id.casefold())
            video = (root / row[path_column]).resolve()
            if not video.is_file() or video.suffix.lower() != ".mp4":
                raise ValueError(f"Missing MP4 for manifest entry {video_id}: {video}")
            entries.append((video_id, video))
    if not entries:
        raise ValueError("Manifest is empty.")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Process a directory with one shared model.")
    parser.add_argument("videos", type=Path)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "code" / "vocabulary.json")
    parser.add_argument("--supported-classes", type=Path, default=ROOT / "supported_classes.json",
                        help="Student-kit class JSON. Class mapping requires the kit format.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--selector", choices=("preserved", "baseline", "duration"), default="preserved")
    parser.add_argument("--unsmoothed", action="store_true")
    parser.add_argument("--top-k", type=int, choices=(1, 3), default=1)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-strength", type=float, default=.1)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--id-column", default="video_id")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--validator", type=Path, help="Organizer's validate.py; run after ALL manifest outputs exist.")
    args = parser.parse_args()
    try:
        supported = load_supported_classes(args.supported_classes)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if supported is not None:
        parser.error(f"Loaded {supported[0]}. Class mapping requires the confirmed student-kit format.")
    if not args.videos.is_dir():
        parser.error("Supply an existing video directory.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.validator and (not args.manifest or not args.validator.is_file()):
        parser.error("Official validation requires an existing validator and an explicit manifest.")
    videos = sorted(p for p in args.videos.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    if args.manifest:
        try:
            entries = manifest_entries(args.manifest, args.videos, args.id_column, args.path_column)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        videos = [path for _, path in entries]
    else:
        entries = [(p.stem, p) for p in videos]
    if not videos:
        parser.error("No MP4 videos found.")
    if len({video_id.casefold() for video_id, _ in entries}) != len(videos):
        parser.error("Duplicate video IDs.")
    expected = {video_id + ".json" for video_id, _ in entries}
    unexpected = {p.name for p in args.outputs.glob("*.json")} - expected
    if unexpected and args.manifest:
        parser.error(f"Output folder contains JSONs outside the manifest: {sorted(unexpected)}")
    adapter = args.adapter
    if adapter is None and not args.no_adapter and args.vocabulary.resolve() == (ROOT / "code/vocabulary.json").resolve():
        adapter = ROOT / "weights/action_adapter.npz"
    if args.no_adapter:
        adapter = None
    scorer = Scorer(args.vocabulary, args.threads, adapter, args.adapter_strength)
    original = Scorer(ROOT / "code/vocabulary-baseline.json", args.threads, shared=scorer) if args.selector == "preserved" else None
    args.outputs.mkdir(parents=True, exist_ok=True)
    for video_id, video in entries:
        duration, _ = probe(video)
        if not 1 <= duration <= 600.05:
            parser.error(f"Unsupported duration for {video.name}")
        raw = frames(video, FRAME_SECONDS)
        scores, idle, embeddings, label = scorer.predict(raw)
        actions, highlights = analyze(scores, idle, embeddings, scorer.actions, duration,
                                      smoothing=not args.unsmoothed)
        if args.selector == "baseline":
            _, highlights = baseline(scores, idle, embeddings, scorer.actions, duration)
        elif original is not None:
            old_scores, old_idle, _, _ = original.predict(raw)
            _, highlights = baseline(old_scores, old_idle, embeddings, original.actions, duration)
        if args.top_k > 1:
            actions = localize(scores, idle, embeddings, scorer.actions, duration,
                               top_k=args.top_k, smoothing=not args.unsmoothed)
        actions, highlights = clean_intervals(actions, highlights, duration)
        result = dict(video_id=video_id, label=label, actions=actions, highlights=highlights)
        check(result, video_id, duration)
        target = args.outputs / (video_id + ".json")
        import os
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    if args.validator:
        subprocess.run([sys.executable, str(args.validator.resolve()), "--dir", str(args.outputs.resolve())],
                       cwd=args.validator.resolve().parent, check=True)
        print(f"Organizer validator exited successfully for {len(entries)} manifest outputs.")


if __name__ == "__main__":
    main()
