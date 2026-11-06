"""Experimental source-clip selection from temporal predictions, under 14.5%."""
import math

import numpy as np
from config import TEMPORAL_SELECTION


def select(actions, verbs, nouns, classes, embeddings, duration, pair_scores=None):
    limit = math.floor(duration * 1000 + 1e-6)
    remaining = limit * int(TEMPORAL_SELECTION.budget_fraction * 1000) // 1000
    verb_ids = {name: i for i, name in enumerate(classes["verb"])}
    noun_ids = {name: i for i, name in enumerate(classes["noun"])}
    candidates = []
    for action in actions:
        pair = (action["verb"], action["noun"])
        # Two-second viewing context, including brief predicted interactions.
        for offset in np.arange(action["start"], action["end"], 2.):
            center = (offset + min(offset + 2., action["end"])) / 2
            start = max(0, min(limit - 2000, round((center - 1.) * 1000)))
            end = min(limit, start + 2000)
            candidates.append((start, end, pair, round(offset * 1000),
                               round(min(offset + 2., action["end"]) * 1000)))
    chosen, covered, prototypes = [], set(), []

    def summary(start, end, pair):
        left, right = int(start // 1000), min(len(verbs), math.ceil(end / 1000))
        indices = np.arange(left, right)
        weights = np.maximum(0, np.minimum((indices + 1) * 1000, end) - np.maximum(indices * 1000, start))
        confidence = (pair_scores[pair][indices] if pair_scores is not None else
                      np.sqrt(verbs[indices, verb_ids[pair[0]]] * nouns[indices, noun_ids[pair[1]]]))
        importance = float(np.average(confidence, weights=weights))
        vector = np.average(embeddings[indices], axis=0, weights=weights)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        return importance, vector

    while remaining > 0 and candidates:
        best = None
        for start, stop, pair, active_start, active_end in candidates:
            end = min(stop, start + remaining)
            if end <= start or any(start < b and end > a for a, b, _ in chosen):
                continue
            importance, vector = summary(start, end, pair)
            redundancy = max((float(vector @ old) for old in prototypes), default=0.)
            retained = max(0, min(end, active_end) - max(start, active_start)) / max(1, active_end - active_start)
            merit = (
                TEMPORAL_SELECTION.novelty_weight * (pair not in covered) * retained / ((end - start) / 1000)
                + importance
                - TEMPORAL_SELECTION.redundancy_penalty * max(0., redundancy)
            )
            if best is None or merit > best[0]:
                best = (merit, start, end, pair, importance, vector, retained)
        if best is None:
            break
        _, start, end, pair, importance, vector, retained = best
        chosen.append((start, end, importance))
        remaining -= end - start
        if retained >= TEMPORAL_SELECTION.retained_pair_min_fraction:
            covered.add(pair)
        prototypes.append(vector)
    return [dict(start=float(a / 1000), end=float(b / 1000), importance=float(np.clip(score, 0, 1)))
            for a, b, score in sorted(chosen)]


def generic_select(actions, scores, vocabulary, embeddings, duration, fallback):
    """Same temporal selector using generic CLIP pair evidence, without TSM."""
    if not actions:
        return fallback
    evidence = np.exp(TEMPORAL_SELECTION.generic_score_temperature * (scores - scores.max(axis=1, keepdims=True)))
    evidence /= evidence.sum(axis=1, keepdims=True)
    pairs = {(a["verb"], a["noun"]): evidence[:, i] for i, a in enumerate(vocabulary)}
    dummy = np.zeros((len(scores), 1))
    return select(actions, dummy, dummy, {"verb": [], "noun": []}, embeddings, duration, pairs)
