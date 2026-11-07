"""Explain missed public-annotation actions; these buckets are not official metrics."""
import csv
import json
from collections import Counter, defaultdict

from evaluate import tiou
from model import ROOT
from temporal_experiment import seconds


def categorize(predictions, truth):
    available = set(range(len(truth)))
    for prediction in sorted(predictions,key=lambda a:a['start']):
        candidates = [i for i in available if (truth[i]['verb'],truth[i]['noun']) == (prediction['verb'],prediction['noun'])]
        best = max(candidates,key=lambda i:tiou(prediction,truth[i]),default=None)
        if best is not None and tiou(prediction,truth[best]) >= .1:
            available.remove(best)
    counts = Counter(matched=len(truth)-len(available))
    examples = defaultdict(list)
    for i in sorted(available):
        action = truth[i]
        nearby = [p for p in predictions if tiou(action,p) >= .1]
        if any((p['verb'],p['noun']) == (action['verb'],action['noun']) for p in nearby):
            bucket = 'duplicate_or_competing_instance'
        elif not nearby:
            bucket = 'no_prediction_with_temporal_overlap'
        elif any(p['noun'] == action['noun'] for p in nearby):
            bucket = 'noun_present_verb_missing'
        elif any(p['verb'] == action['verb'] for p in nearby):
            bucket = 'verb_present_noun_missing'
        else:
            bucket = 'both_labels_missing_at_time'
        counts[bucket] += 1
        if len(examples[bucket]) < 2:
            examples[bucket].append(dict(ground_truth=action,predictions=sorted(nearby,key=lambda p:tiou(p,action),reverse=True)[:2]))
    return dict(counts), dict(examples)


def main():
    rows = json.loads((ROOT/'docs/recognition/fresh-comparison-results.json').read_text())
    classes = json.loads((ROOT/'weights/temporal/classes.json').read_text())
    with (ROOT/'work/recognition/EPIC_100_validation.csv').open(encoding='utf-8-sig') as f:
        annotations = list(csv.DictReader(f))
    results,total = [],Counter()
    for row in (r for r in rows if r['profile']=='scene_joint'):
        truth = [dict(verb=classes['verb'][int(r['verb_class'])],noun=classes['noun'][int(r['noun_class'])],
                      start=seconds(r['start_timestamp']),end=seconds(r['stop_timestamp']))
                 for r in annotations if r['video_id']==row['video_id']]
        counts,examples = categorize(row['predictions'],truth)
        assert sum(counts.values()) == len(truth)
        total.update(counts)
        results.append(dict(video_id=row['video_id'],counts=counts,examples=examples))
    report = dict(scope='Reused local three-video comparison. Greedy matching at tIoU .1; descriptive error buckets, not Samsung scores.',
                  total=dict(total),results=results)
    target = ROOT/'work/jury-review/errors.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['total'],indent=2))


if __name__ == '__main__':main()
