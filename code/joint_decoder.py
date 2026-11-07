"""Joint verb/noun decoding using training-only associations and temporal continuity."""
import numpy as np


def joint_segments(verbs, nouns, classes, duration, association, strength, switch_cost):
    from temporal_model import segments
    verbs, nouns = np.asarray(verbs), np.asarray(nouns)
    association = np.asarray(association)
    if association.shape != (verbs.shape[1], nouns.shape[1]):
        raise ValueError("Decoder class dimensions do not match the model")
    if strength < 0 or switch_cost < 0 or not np.isfinite(association).all():
        raise ValueError("Invalid joint decoder settings")
    scores = (np.log(np.maximum(verbs, 1e-12))[:, :, None]
              + np.log(np.maximum(nouns, 1e-12))[:, None, :]
              + strength * association[None, :, :]).reshape(len(verbs), -1)
    # Exact Viterbi for a uniform switch cost: stay, or switch from the best state.
    previous = scores[0].copy()
    back = np.empty(scores.shape, dtype=np.int32)
    states = np.arange(scores.shape[1], dtype=np.int32)
    for t in range(1, len(scores)):
        best = int(previous.argmax())
        stay = previous >= previous[best] - switch_cost
        back[t] = np.where(stay, states, best)
        previous = scores[t] + np.where(stay, previous, previous[best] - switch_cost)
    path = [int(previous.argmax())]
    for t in range(len(scores) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return segments(((i // nouns.shape[1], i % nouns.shape[1]) for i in path), classes, duration)


def load_profile(path):
    with np.load(path, allow_pickle=False) as data:
        return data['association'], float(data['strength']), float(data['switch_cost'])
