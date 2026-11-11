import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from demo import build_demo
from media import frames, probe, render
from model import ROOT, Scorer, load_supported_classes
from pipeline import PipelineOptions, PipelineState, infer_video
from selection import FRAME_SECONDS
from structured_log import log_event
from runtime_profile import apply_profile
from submission import make_submission


def main():
    parser = argparse.ArgumentParser(description="Extract action highlights from a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "code" / "vocabulary.json")
    parser.add_argument("--supported-classes", type=Path, default=ROOT / "code/supported_classes.json",
                        help="Student-kit class JSON. Class mapping requires the kit format.")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
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
    parser.add_argument("--demo", type=Path, metavar="DIR")
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
        duration, audio = probe(source)
        if not 1 <= duration:
            raise ValueError("This prototype requires a video duration of at least one second.")
        adapter = args.adapter
        if adapter is None and args.adapter_strength > 0 and not args.no_adapter and args.vocabulary.resolve() == (ROOT / "code/vocabulary.json").resolve():
            adapter = ROOT / "weights/action_adapter.npz"
        if args.no_adapter:
            adapter = None
        scorer = Scorer(args.vocabulary, args.threads, adapter, args.adapter_strength, supported=supported)
        original = Scorer(ROOT / "code/vocabulary-baseline.json", args.threads, shared=scorer) if args.selector == "preserved" else None
        raw = frames(source, FRAME_SECONDS)
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
        inferred = infer_video(source, duration, raw, options, PipelineState(scorer=scorer, original=original))
        log_event(
            "route_selected",
            route=inferred.route,
            temporal_model=bool(args.temporal_model),
            kitchen_gate_accepted=inferred.kitchen_gate_accepted,
            profile=args.profile,
            selector=args.selector,
            kitchen_routing=args.kitchen_routing,
        )
        result = make_submission(source.stem, duration, inferred, supported[1])
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Track 4: {exc}\n")
    if args.demo:
        try:
            render(source, reel, [dict(start=h["segment"][0], end=h["segment"][1]) for h in result["highlights"]], audio)
            build_demo(page, source, reel, dict(video_id=source.stem, label=inferred.label, actions=inferred.actions, highlights=inferred.highlights), duration)
        except (OSError, ValueError, RuntimeError) as exc:
            log_event("demo_failed", reason=str(exc), inference_json_written=str(output))
    selected = sum(item["segment"][1] - item["segment"][0] for item in result["highlights"])
    log_event(
        "inference_complete",
        actions=len(inferred.actions),
        selected_seconds=round(selected, 6),
        duration_seconds=round(duration, 6),
        selected_fraction=round(selected / duration, 6),
        elapsed_seconds=round(perf_counter() - started, 6),
    )


if __name__ == "__main__":
    main()
