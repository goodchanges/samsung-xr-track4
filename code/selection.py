import math

import numpy as np

FRAME_SECONDS = 1
MILLISECONDS = 1000


def unit_scale(values):
    span = float(np.ptp(values))
    return (values - values.min()) / span if span > 1e-6 else np.zeros_like(values)


def analyze(scores, idle, embeddings, vocabulary, duration, smoothing=True):
    end = math.floor(duration * MILLISECONDS + 1e-6)
    step = FRAME_SECONDS * MILLISECONDS
    budget = end * 15 // 100
    length = min(len(scores), math.ceil(end / step))
    scores, idle, embeddings = scores[:length], idle[:length], embeddings[:length]
    before = np.maximum(np.arange(length) - 1, 0)
    after = np.minimum(np.arange(length) + 1, length - 1)
    smooth = 0.6 * scores + 0.2 * (scores[before] + scores[after])
    if not smoothing:
        smooth = scores
    labels = smooth.argmax(axis=1)
    margin = smooth.max(axis=1) - idle
    change = np.clip(1 - (embeddings[before] * embeddings[after]).sum(axis=1), 0, 1)
    valid = (margin > 0.005) & ((change > 0.012) | (margin > 0.045))
    activity = 0.7 * np.clip(margin / 0.10, 0, 1) + 0.3 * unit_scale(change)

    actions = []
    candidates = []
    i = 0
    while i < length:
        if not valid[i]:
            i += 1
            continue
        label = int(labels[i])
        j = i + 1
        while j < length and valid[j] and labels[j] == label:
            j += 1
        source = vocabulary[label]
        actions.append({
            "verb": source["verb"], "noun": source["noun"],
            "start": i * FRAME_SECONDS, "end": min(j * step, end) / MILLISECONDS,
        })
        for start in range(i, j, 4):
            stop = min(j, start + 4)
            candidates.append((start, stop, (source["verb"], source["noun"])))
        i = j

    # Include visually active regions where the vocabulary lacks a convincing label.
    if float(activity.max()) > 0.015:
        candidates.extend((i, min(i + 3, length), None) for i in range(0, length, 3)
                          if float(activity[i:i + 3].max()) > 0.015)
    # Prefix sums make each candidate's retained-frame summary constant time.
    embedding_prefix = np.vstack([np.zeros(embeddings.shape[1]), np.cumsum(embeddings, axis=0)])
    activity_prefix = np.r_[0.0, np.cumsum(activity)]
    summaries = {}

    def summarize(start_index, stop_ms):
        key = (start_index, stop_ms)
        if key not in summaries:
            full_end, tail_ms = divmod(stop_ms, step)
            full_end = min(full_end, length)
            count = full_end - start_index
            vector = embedding_prefix[full_end] - embedding_prefix[start_index]
            score = activity_prefix[full_end] - activity_prefix[start_index]
            if tail_ms and full_end < length:
                fraction = tail_ms / step
                vector = vector + fraction * embeddings[full_end]
                score += fraction * activity[full_end]
                count += fraction
            vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
            summaries[key] = (vector, float(score / max(count, 1e-8)))
        return summaries[key]

    for start_index, stop_index, _ in candidates:
        summarize(start_index, min(stop_index * step, end))
    chosen = []
    covered = set()
    prototypes = []
    remaining = budget
    while remaining > 0 and candidates:
        best = None
        for i, j, pair in candidates:
            start = i * step
            stop = min(j * step, end, start + remaining)
            if stop <= start or any(start < b and stop > a for a, b, _ in chosen):
                continue
            vector, base = summarize(i, stop)
            repeat = max((float(vector @ other) for other in prototypes), default=0.0)
            novel = pair is not None and pair not in covered
            seconds = (stop - start) / MILLISECONDS
            merit = (0.8 * novel + seconds * (base + 0.08 * (pair is not None) - 0.22 * max(repeat, 0))) / seconds
            if best is None or merit > best[0]:
                best = (merit, start, stop, pair, vector, base)
        if best is None:
            break
        _, start, stop, pair, vector, base = best
        chosen.append((start, stop, round(base, 4)))
        covered.add(pair)
        prototypes.append(vector)
        remaining -= stop - start
    if not chosen and budget:
        start = max(0, (end - budget) // 2)
        chosen.append((start, start + budget, 0.0))
    highlights = [
        {"start": start / MILLISECONDS, "end": stop / MILLISECONDS, "importance": importance}
        for start, stop, importance in sorted(chosen)
    ]
    return actions, highlights
