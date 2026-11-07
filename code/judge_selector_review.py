"""Compare current summarizers with frozen scene/joint recognition. Local metrics only."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from evaluate import metrics
from media import frames, probe
from model import ROOT, Scorer
from pipeline import PipelineOptions, PipelineState, infer_video
from refine_temporal import DEVELOPMENT
from temporal_experiment import seconds
from temporal_model import TemporalRecognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('development', 'comparison'))
    args = parser.parse_args()
    folder = ROOT/'work/jury-review'
    folder.mkdir(parents=True, exist_ok=True)
    freeze = folder/'selector-frozen.json'
    if args.split == 'development' and freeze.exists():
        parser.error('Selector already frozen; do not overwrite.')
    if args.split == 'comparison' and (folder/'comparison.json').exists():
        parser.error('Comparison already completed; do not retune against it.')
    if args.split == 'comparison':
        frozen = json.loads(freeze.read_text())
    with (ROOT/'work/recognition/EPIC_100_validation.csv').open(encoding='utf-8-sig') as f:
        annotations = list(csv.DictReader(f))
    classes = json.loads((ROOT/'weights/temporal/classes.json').read_text())
    scorer = Scorer(ROOT/'code/vocabulary.json',4)
    state = PipelineState(scorer,Scorer(ROOT/'code/vocabulary-baseline.json',4,shared=scorer),
                          TemporalRecognizer(ROOT/'weights/temporal/tsm_rgb.ckpt'))
    ids = DEVELOPMENT if args.split == 'development' else ['P01_12','P06_12','P11_22']
    rows = []
    for video_id in ids:
        video = ROOT/('work/temporal' if args.split=='development' else 'work/fresh-comparison')/(video_id+'.mp4')
        duration,_ = probe(video)
        raw = frames(video,1)
        truth = dict(actions=[dict(verb=classes['verb'][int(r['verb_class'])],noun=classes['noun'][int(r['noun_class'])],
                     start=seconds(r['start_timestamp']),end=seconds(r['stop_timestamp'])) for r in annotations if r['video_id']==video_id])
        budget = int(np.floor(duration*1000+1e-6))*145//1000/1000
        width = budget/10
        uniform = [dict(start=float((i+.5)*duration/10-width/2),end=float((i+.5)*duration/10+width/2),importance=0.) for i in range(10)]
        for selector in ('preserved','temporal'):
            options = PipelineOptions(selector,False,1,4,ROOT/'weights/temporal/tsm_rgb.ckpt',4,
                                      ROOT/'weights/temporal_decoder.npz','scene')
            result = infer_video(video,duration,raw,options,state)
            for name,highlights in [(selector,result.highlights)] + ([('uniform',uniform)] if selector=='preserved' else []):
                measured = metrics(result.actions,highlights,result.label,truth,duration)
                rows.append(dict(video_id=video_id,selector=name,route=result.route,highlights=highlights,metrics=measured))
                print(video_id,name,measured['coverage_at_half_action'],flush=True)
    summary = {name:float(np.mean([r['metrics']['coverage_at_half_action'] for r in rows if r['selector']==name]))
               for name in ('preserved','temporal','uniform')}
    if args.split == 'development':
        best = max(('preserved','temporal'),key=lambda name:summary[name])
        freeze.write_text(json.dumps(dict(selected=best,context=4,recognition='scene/joint',development_ids=ids,
             comparison_ids=['P01_12','P06_12','P11_22'],summary=summary,frozen_at=datetime.now(timezone.utc).isoformat(),
             recognition_profile_sha256=hashlib.sha256((ROOT/'weights/temporal_decoder.npz').read_bytes()).hexdigest()),indent=2))
    (folder/(args.split+'.json')).write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':main()
