"""Fit pair statistics on public training data; select on development, compare once."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate import metrics
from joint_decoder import joint_segments
from media import probe
from model import ROOT
from refine_temporal import DEVELOPMENT, EVALUATION, mean_f1
from temporal_experiment import seconds
from temporal_model import TemporalRecognizer, decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('development', 'evaluation'))
    args = parser.parse_args()
    folder = ROOT / 'work/joint-decoder'
    folder.mkdir(parents=True, exist_ok=True)
    profile = ROOT / 'weights/temporal_decoder.npz'
    frozen = folder / 'frozen.json'
    if args.split == 'development' and frozen.exists():
        parser.error('Development selection already frozen; do not retune after evaluation.')
    classes = json.loads((ROOT / 'weights/temporal/classes.json').read_text())
    metadata = ROOT / 'work/recognition'
    def read(name):
        with (metadata / name).open(encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    training = read('EPIC_100_train.csv')
    assert not set(DEVELOPMENT + EVALUATION) & {r['video_id'] for r in training}
    annotations = read('EPIC_100_validation.csv')
    if args.split == 'development':
        counts = np.ones((97, 300), dtype=float)  # Add-one smoothing retains unseen pairs.
        for row in training:
            counts[int(row['verb_class']), int(row['noun_class'])] += 1
        # Log association, rather than a raw frequency prior favoring common labels.
        association = np.log(counts / counts.sum(axis=0, keepdims=True)) - np.log(counts.sum(axis=1, keepdims=True) / counts.sum())
        configs = [dict(strength=s, switch_cost=c) for s in (0., .5, 1.) for c in (0., .5, 1.)]
        ids = DEVELOPMENT
    else:
        settings = json.loads(frozen.read_text())
        with np.load(profile, allow_pickle=False) as saved:
            association = saved['association']
        assert hashlib.sha256(profile.read_bytes()).hexdigest() == settings['profile_sha256']
        configs = [dict(strength=0., switch_cost=0.), settings['selected']]
        ids = EVALUATION
    model = TemporalRecognizer(ROOT / 'weights/temporal/tsm_rgb.ckpt')
    rows = []
    for video_id in ids:
        video = ROOT / 'work/temporal' / (video_id + '.mp4')
        duration, _ = probe(video)
        verbs, nouns = model.score_video(video, duration, 4)
        truth = dict(actions=[dict(verb=classes['verb'][int(r['verb_class'])], noun=classes['noun'][int(r['noun_class'])],
                                   start=seconds(r['start_timestamp']), end=seconds(r['stop_timestamp']))
                              for r in annotations if r['video_id'] == video_id])
        for config in configs:
            actions = joint_segments(verbs, nouns, classes, duration, association, **config)
            if config == dict(strength=0., switch_cost=0.):
                assert actions == decode(verbs, nouns, classes, duration)
            rows.append(dict(video_id=video_id, config=config, predictions=actions, metrics=metrics(actions, [], None, truth, duration)))
        print(video_id, 'completed', flush=True)
    summary = [dict(config=c, f1=mean_f1([r for r in rows if r['config'] == c])) for c in configs]
    if args.split == 'development':
        best = max(summary, key=lambda r:r['f1'])['config']
        np.savez_compressed(profile, association=association, **best)
        frozen.write_text(json.dumps(dict(selected=best, summary=summary, development=DEVELOPMENT, evaluation=EVALUATION,
                         fit_video_ids=sorted({r['video_id'] for r in training}), fit_rows=len(training),
                         annotation_revision='ea8b40457a400c3fffa1c7f406ef3dc169cc2522',
                         checkpoint_sha256=model.checkpoint_hash,
                         fitted_on='EPIC_100_train.csv only', training_sha256=hashlib.sha256((metadata/'EPIC_100_train.csv').read_bytes()).hexdigest(),
                         profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), frozen_at=datetime.now(timezone.utc).isoformat()), indent=2))
    (folder / (args.split + '.json')).write_text(json.dumps(dict(summary=summary, results=rows), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
