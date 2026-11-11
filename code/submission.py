"""Serialize the supplied kit format; internal inference keeps its interval format."""
from math import floor, ceil
from structured_log import log_event

TEAM_ID = "VelloreInstituteOfTechnologyVellore_VIT"


def failure_submission(video_id, duration=0.0):
    highlights = []
    if duration >= 4:
        highlights = [dict(segment=[0.0, min(4.0, float(duration))])]
    return dict(team_id=TEAM_ID, video_id=video_id, duration_sec=float(duration),
                label=dict(name='other', score=0.0), actions=[], highlights=highlights)


def check_submission(document):
    """Local assertions from the guide. This is not the organizer validator."""
    import json
    import math
    from pathlib import Path
    classes = json.loads(Path(__file__).with_name('supported_classes.json').read_text())
    assert set(document) == {'team_id', 'video_id', 'duration_sec', 'label', 'actions', 'highlights'}
    assert document['team_id'] and document['video_id']
    duration = document['duration_sec']
    assert math.isfinite(duration) and duration > 0
    assert document['label']['name'] in classes['labels']
    assert 0 <= document['label']['score'] <= 1
    assert len(document['actions']) <= 2000
    for action in document['actions']:
        assert action['verb'] in classes['verbs'] and action['noun'] in classes['nouns']
        assert 0 <= action['score'] <= 1
        assert 0 <= action['segment'][0] < action['segment'][1] <= duration + 0.5
    previous = 0.0
    total = 0.0
    assert document['highlights']
    for row in document['highlights']:
        start, end = row['segment']
        assert previous <= start < end <= duration + 0.5
        assert end - start >= 4 - 1e-8
        total += end - start
        previous = end
    assert total <= max(4.0, duration * 0.145) + 0.01


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
