from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localization import localize
from output import clean_intervals
from selection import analyze
from selection_baseline import analyze as baseline
from structured_log import log_event


@dataclass(frozen=True)
class PipelineOptions:
    selector: str
    unsmoothed: bool
    top_k: int
    threads: int
    temporal_model: Path | None
    temporal_context: int


@dataclass
class PipelineState:
    scorer: Any
    original: Any = None
    temporal: Any = None


@dataclass(frozen=True)
class InferenceResult:
    label: str
    actions: list[dict]
    highlights: list[dict]
    route: str
    kitchen_gate_accepted: bool
    kitchen_margin: float
    kitchen_frame_fraction: float


def infer_video(source, duration, raw, options: PipelineOptions, state: PipelineState):
    scores, idle, embeddings, label = state.scorer.predict(raw)
    actions, highlights = analyze(
        scores, idle, embeddings, state.scorer.actions, duration, smoothing=not options.unsmoothed
    )
    if options.selector == "baseline":
        _, highlights = baseline(scores, idle, embeddings, state.scorer.actions, duration)
    elif options.selector == "preserved":
        old_scores, old_idle, _, _ = state.original.predict(raw)
        _, highlights = baseline(old_scores, old_idle, embeddings, state.original.actions, duration)
    if options.top_k > 1:
        actions = localize(
            scores, idle, embeddings, state.scorer.actions, duration,
            top_k=options.top_k, smoothing=not options.unsmoothed
        )
    kitchen, margin, fraction = state.scorer.kitchen_confidence(embeddings)
    route = "clip"
    if options.temporal_model:
        log_event(
            "kitchen_gate",
            accepted=bool(kitchen),
            margin=round(float(margin), 6),
            frame_fraction=round(float(fraction), 6),
        )
    if options.temporal_model and not kitchen:
        from temporal_selection import generic_select
        highlights = generic_select(actions, scores, state.scorer.actions, embeddings, duration, highlights)
        route = "generic_clip_temporal_selector"
    if options.temporal_model and kitchen:
        route = "kitchen_tsm"
        if state.temporal is None:
            try:
                from temporal_model import TemporalRecognizer
            except ImportError as exc:
                raise ValueError(
                    "Install optional model dependencies: python -m pip install -r requirements-temporal.txt"
                ) from exc
            state.temporal = TemporalRecognizer(options.temporal_model, options.threads)
        actions = state.temporal.recognize(
            source, duration, options.temporal_model.parent / "classes.json", options.temporal_context
        )
        if options.selector == "temporal":
            from temporal_selection import select
            import json

            verbs, nouns = state.temporal.score_video(source, duration, options.temporal_context)
            classes = json.loads((options.temporal_model.parent / "classes.json").read_text(encoding="utf-8"))
            highlights = select(actions, verbs, nouns, classes, embeddings, duration)
    actions, highlights = clean_intervals(actions, highlights, duration)
    return InferenceResult(
        label=label,
        actions=actions,
        highlights=highlights,
        route=route,
        kitchen_gate_accepted=bool(kitchen),
        kitchen_margin=float(margin),
        kitchen_frame_fraction=float(fraction),
    )
