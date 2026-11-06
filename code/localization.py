"""Localize concurrent action hypotheses without forcing one action per frame."""
import numpy as np


def localize(scores, idle, embeddings, vocabulary, duration, top_k=3, smoothing=True):
    count = min(len(scores), int(np.ceil(duration)))
    scores, idle, embeddings = scores[:count], idle[:count], embeddings[:count]
    before = np.maximum(np.arange(count) - 1, 0)
    after = np.minimum(np.arange(count) + 1, count - 1)
    if smoothing:
        scores = .6 * scores + .2 * (scores[before] + scores[after])
    change = np.clip(1 - (embeddings[before] * embeddings[after]).sum(axis=1), 0, 1)
    ranks = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    active = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(active, ranks, True, axis=1)
    margin = scores - idle[:, None]
    # Additional hypotheses must be close to the best action and beat idle.
    active &= scores >= scores.max(axis=1, keepdims=True) - .025
    active &= (margin > .005) & ((change[:, None] > .012) | (margin > .045))
    actions = []
    for label in np.flatnonzero(active.any(axis=0)):
        transitions = np.diff(np.r_[False, active[:, label], False].astype(int))
        for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            actions.append(dict(verb=vocabulary[label]["verb"], noun=vocabulary[label]["noun"],
                                start=int(start), end=min(float(end), duration)))
    return sorted(actions, key=lambda row: (row["start"], row["end"], row["verb"], row["noun"]))
