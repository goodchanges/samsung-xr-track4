import argparse
import json
import csv
import subprocess
import sys
import os
from time import perf_counter
from pathlib import Path

from check_output import check
from media import frames, probe
from model import ROOT, Scorer, load_supported_classes
from pipeline import PipelineOptions, PipelineState, infer_video
from selection import FRAME_SECONDS
from structured_log import log_event
from runtime_profile import apply_profile
from submission import make_submission, failure_submission, check_submission


def manifest_entries(manifest, root, id_column="video_id", path_column="path", available_only=False):
    """Use explicit manifest IDs, never guess dataset paths or silently skip rows."""
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if id_column not in (reader.fieldnames or []):
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
            video = (root / (row[path_column] if path_column in row else video_id + ".mp4")).resolve()
            if video.suffix.lower() != ".mp4" or (not video.is_file() and not available_only):
                raise ValueError(f"Missing MP4 for manifest entry {video_id}: {video}")
            entries.append((video_id, video))
    if not entries:
        raise ValueError("Manifest is empty.")
    return entries


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Process a directory with one shared model.")
    parser.add_argument("videos", type=Path)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "code" / "vocabulary.json")
    parser.add_argument("--supported-classes", type=Path, default=ROOT / "code/supported_classes.json",
                        help="Student-kit class JSON. Class mapping requires the kit format.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--selector", choices=("preserved", "baseline", "duration", "temporal"))
    parser.add_argument("--profile", choices=("clip", "improved"), help="Deployment preset; otherwise use packaged preset or CLIP.")
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
    parser.add_argument("--temporal-decoder", type=Path, help="Optional measured joint decoder profile; requires four-second TSM contexts.")
    parser.add_argument("--kitchen-routing", choices=("category", "scene"), help="Experimental scene-based kitchen gate, or existing category gate.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--available-only", action="store_true", help="Process available manifest paths and report missing files without generating placeholder outputs for them.")
    parser.add_argument("--id-column", default="video_id")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--validator", type=Path, help="Organizer's validate.py; run after ALL manifest outputs exist.")
    parser.add_argument("--report", type=Path, default=ROOT / "work/batch-report.json",
                        help="Per-video status report, stored outside submission outputs.")
    args = parser.parse_args()
    try:
        apply_profile(args, ROOT)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.selector == "temporal" and not args.temporal_model:
        parser.error("--selector temporal requires --temporal-model.")
    context = args.temporal_context or (2 if args.selector == "temporal" else 4)
    if args.temporal_decoder and (not args.temporal_model or context != 4):
        parser.error("--temporal-decoder requires --temporal-model and --temporal-context 4.")
    try:
        supported = load_supported_classes(args.supported_classes)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not args.videos.is_dir():
        parser.error("Supply an existing video directory.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.available_only and not args.manifest:
        parser.error("--available-only requires --manifest.")
    if args.validator and (not args.manifest or not args.validator.is_file()):
        parser.error("Official validation requires an existing validator and an explicit manifest.")
    if args.report.resolve().is_relative_to(args.outputs.resolve()):
        parser.error("Keep the batch status report outside the submission output directory.")
    videos = sorted(p for p in args.videos.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    if args.manifest:
        try:
            entries = manifest_entries(args.manifest, args.videos, args.id_column, args.path_column, args.available_only)
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
    durations = {}
    if args.manifest:
        with args.manifest.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if 'duration_sec' in row:
                    durations[row[args.id_column].strip()] = float(row['duration_sec'])
    adapter = args.adapter
    if adapter is None and args.adapter_strength > 0 and not args.no_adapter and args.vocabulary.resolve() == (ROOT / "code/vocabulary.json").resolve():
        adapter = ROOT / "weights/action_adapter.npz"
    if args.no_adapter:
        adapter = None
    state = None
    options = PipelineOptions(
        selector=args.selector,
        unsmoothed=args.unsmoothed,
        top_k=args.top_k,
        threads=args.threads,
        temporal_model=args.temporal_model,
        temporal_context=context,
        temporal_decoder=args.temporal_decoder,
        kitchen_routing=args.kitchen_routing,
    )
    args.outputs.mkdir(parents=True, exist_ok=True)
    report = []
    for video_id, video in entries:
        started = perf_counter()
        duration = 0.0
        record = dict(video_id=video_id, source=str(video), profile=args.profile)
        if not video.is_file() and args.available_only:
            record.update(status='missing', reason='Source MP4 is unavailable', elapsed_seconds=0.0)
            report.append(record)
            write_json(args.report, report)
            continue
        try:
            duration, _ = probe(video)
            if not 1 <= duration:
                raise ValueError(f"Unsupported duration for {video.name}")
            if video_id in durations and abs(duration - durations[video_id]) > 0.5 + 1e-8:
                raise ValueError(f"Video duration {duration} differs from manifest {durations[video_id]} by more than 0.5 seconds")
            if state is None:
                scorer = Scorer(args.vocabulary, args.threads, adapter, args.adapter_strength, supported=supported)
                original = Scorer(ROOT / "code/vocabulary-baseline.json", args.threads, shared=scorer) if args.selector == "preserved" else None
                state = PipelineState(scorer=scorer, original=original)
            raw = frames(video, FRAME_SECONDS)
            inferred = infer_video(video, duration, raw, options, state)
            log_event("batch_video_route", video_id=video_id, route=inferred.route,
                      profile=args.profile, selector=args.selector, kitchen_routing=args.kitchen_routing)
            result = make_submission(video_id, duration, inferred, supported[1])
            check(result, video_id, duration)
            warnings = check_submission(result)
            record.update(status='unresolved' if warnings else 'ok', duration_sec=duration,
                          route=inferred.route, warnings=warnings)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"Batch failure [{video_id}]: {reason}", file=sys.stderr)
            log_event("batch_video_failed", video_id=video_id, reason=reason)
            result = failure_submission(video_id, duration)
            record.update(status='failed', reason=reason)
        target = args.outputs / (video_id + ".json")
        try:
            write_json(target, result)
            record['output'] = str(target)
        except OSError as exc:
            record.update(status='failed', reason=f"Output write failed: {exc}")
            print(f"Batch failure [{video_id}]: {record['reason']}", file=sys.stderr)
        record['elapsed_seconds'] = round(perf_counter() - started, 6)
        report.append(record)
        write_json(args.report, report)
    counts = {status: sum(row['status'] == status for row in report)
              for status in ('ok', 'failed', 'missing', 'unresolved')}
    print(json.dumps(dict(total=len(report), **counts, report=str(args.report))))
    if counts['failed'] or counts['missing'] or counts['unresolved']:
        parser.exit(1, f"Batch incomplete; see {args.report}.\n")
    if args.validator:
        subprocess.run([sys.executable, str(args.validator.resolve()), "--dir", str(args.outputs.resolve())],
                       cwd=args.validator.resolve().parent, check=True)
        print(f"Organizer validator exited successfully for {len(entries)} manifest outputs.")


if __name__ == "__main__":
    main()
