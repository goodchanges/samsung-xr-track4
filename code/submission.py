"""Serialize the supplied kit format; internal inference keeps its interval format."""
from math import floor, ceil
from structured_log import log_event

TEAM_ID = "VIT"


def check_submission(document, classes=None):
    """Check the supplied guide and our 14.5% safety margin; return unresolved warnings."""
    import json
    import math
    from pathlib import Path
    if classes is None:
        classes = json.loads(Path(__file__).with_name('supported_classes.json').read_text())

    def fields(value, expected, location):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"{location}: expected exactly {', '.join(expected)}")

    def number(value, location):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{location}: expected a finite number (not a boolean)")
        return value

    def score(value, location):
        if not 0 <= number(value, location) <= 1:
            raise ValueError(f"{location}: must be in [0, 1]")

    def member(value, names, location):
        if not isinstance(value, str) or value not in names:
            raise ValueError(f"{location}: unsupported class {value!r}")

    def segment(row, location):
        value = row['segment']
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{location}.segment: expected [start, end]")
        start, end = [number(v, location + '.segment') for v in value]
        if any(abs(v - round(v, 2)) > 1e-8 for v in value):
            raise ValueError(f"{location}.segment: use at most two decimal places")
        if not 0 <= start < end <= duration + 0.5:
            raise ValueError(f"{location}.segment: outside [0, duration_sec + 0.5] or empty")
        return start, end

    fields(document, ('team_id', 'video_id', 'duration_sec', 'label', 'actions', 'highlights'), 'output')
    for key in ('team_id', 'video_id'):
        if not isinstance(document[key], str) or not document[key].strip():
            raise ValueError(f"{key}: expected a nonempty string")
    duration = number(document['duration_sec'], 'duration_sec')
    if duration <= 0:
        raise ValueError('duration_sec: must be positive')
    fields(document['label'], ('name', 'score'), 'label')
    member(document['label']['name'], classes['labels'], 'label.name')
    score(document['label']['score'], 'label.score')
    if not isinstance(document['actions'], list) or len(document['actions']) > 2000:
        raise ValueError('actions: expected an array of at most 2000 items')
    for i, action in enumerate(document['actions']):
        location = f'actions[{i}]'
        fields(action, ('segment', 'verb', 'noun', 'score'), location)
        member(action['verb'], classes['verbs'], location + '.verb')
        member(action['noun'], classes['nouns'], location + '.noun')
        score(action['score'], location + '.score')
        segment(action, location)
    if not isinstance(document['highlights'], list) or not document['highlights']:
        raise ValueError('highlights: expected a nonempty array')
    previous = total = 0.0
    for i, row in enumerate(document['highlights']):
        location = f'highlights[{i}]'
        fields(row, ('segment',), location)
        start, end = segment(row, location)
        if start < previous:
            raise ValueError(f'{location}: highlights must be sorted and nonoverlapping')
        if end - start < 4 - 1e-8:
            raise ValueError(f'{location}: must be at least four seconds')
        total += end - start
        previous = end
    if total > duration * 0.145 + 1e-8:
        if duration * 0.15 < 4 and abs(total - 4) <= 1e-8:
            return ['UNRESOLVED: four-second highlight exceeds even the 15% budget; organizer clarification required']
        if total <= duration * .15 + 1e-8 and abs(total - 4) <= 1e-8:
            return []
        raise ValueError(f'highlights: {total:.2f}s exceeds the local 14.5% limit ({duration * .145:.4f}s)')
    return []


def make_submission(video_id, duration, inferred, classes):
    actions = []
    for row in inferred.actions:
        start = ceil(max(0.0, row['start']) * 100) / 100
        end = floor(min(duration, row['end']) * 100) / 100
        if start < end and row['verb'] in classes['verbs'] and row['noun'] in classes['nouns']:
            actions.append(dict(segment=[start, end], verb=row['verb'], noun=row['noun'], score=float(row.get('score', 0.0))))
    budget = floor(duration * 14.5 + 1e-8)
    chosen = []
    for row in sorted(inferred.highlights, key=lambda h: -h['importance']):
        start = ceil(max(0.0, row['start']) * 100)
        end = min(floor(duration * 100), max(start + 400, floor(row['end'] * 100)), start + budget)
        if any(start < b and end > a for a, b in chosen):
            continue
        if end - start >= 400:
            chosen.append((start, end))
            budget -= end - start
    if not chosen:
        length = min(floor(duration * 100), max(400, budget))
        best = max(inferred.highlights, key=lambda h: h['importance'], default={'start': 0.0})
        start = min(max(0, ceil(best['start'] * 100)), floor(duration * 100) - length)
        chosen = [(start, start + length)]
        if length > floor(duration * 14.5 + 1e-8):
            log_event('highlight_rule_conflict', duration=duration, reason='Four-second minimum exceeds 14.5% budget; prioritize minimum segment length pending organizer clarification.')
    return dict(team_id=TEAM_ID, video_id=video_id, duration_sec=round(float(duration), 2),
                label=dict(name=inferred.label if inferred.label in classes['labels'] else 'other', score=float(inferred.label_score)),
                actions=actions[:2000], highlights=[dict(segment=[a / 100, b / 100]) for a, b in sorted(chosen)])
