"""Freeze, download and evaluate a new three-participant validation comparison once."""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from urllib.request import urlopen

import numpy as np

from evaluate import metrics
from media import frames, probe
from model import ROOT, Scorer
from pipeline import PipelineOptions, PipelineState, infer_video
from temporal_experiment import seconds
from temporal_model import TemporalRecognizer, decode
from scene_gate import PROMPTS, MARGIN, FRAME_FRACTION

IDS = ['P01_12', 'P06_12', 'P11_22']
FOLDER = ROOT / 'work/fresh-comparison'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'evaluate'))
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    freeze = FOLDER / 'frozen.json'
    if args.stage == 'prepare':
        if freeze.exists():
            parser.error('Comparison already frozen; do not overwrite.')
        provenance = json.loads((ROOT/'docs/recognition/temporal-refinement-provenance.json').read_text())
        record = dict(ids=IDS, selection='First eligible sorted video IDs from distinct participants outside the previous six-video comparison; last annotated time 120-240 seconds. No recognition scores inspected.',
                      prompts=PROMPTS, margin=MARGIN, frame_fraction=FRAME_FRACTION,
                      decoder_sha256=digest(ROOT/'weights/temporal_decoder.npz'), context=4,
                      checkpoint_sha256=digest(ROOT/'weights/temporal/tsm_rgb.ckpt'),
                      validation_sha256=digest(ROOT/'work/recognition/EPIC_100_validation.csv'),
                      model_sources=provenance, files={}, frozen_at=datetime.now(timezone.utc).isoformat(),
                      implementation_sha256={name:digest(ROOT/'code'/name) for name in ('scene_gate.py','pipeline.py','joint_decoder.py')})
        freeze.write_text(json.dumps(record,indent=2))
        for video_id in IDS:
            target = FOLDER / (video_id+'.mp4')
            url = f"{provenance['mirror']}/resolve/{provenance['mirror_revision']}/videos/{video_id}.MP4"
            if not target.exists():
                temporary = target.with_suffix('.partial')
                try:
                    with urlopen(url,timeout=90) as response, temporary.open('wb') as out:
                        while chunk := response.read(1024*1024): out.write(chunk)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            record['files'][video_id] = dict(url=url,sha256=digest(target),bytes=target.stat().st_size)
            freeze.write_text(json.dumps(record,indent=2))
            print('Downloaded',video_id,flush=True)
        return
    if (FOLDER/'complete.json').exists():
        parser.error('This comparison has already completed; do not retune on it.')
    record = json.loads(freeze.read_text())
    assert record['decoder_sha256'] == digest(ROOT/'weights/temporal_decoder.npz')
    for name,expected in record['implementation_sha256'].items():
        assert digest(ROOT/'code'/name) == expected
    def read(name):
        with (ROOT/'work/recognition'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    assert not set(IDS) & {r['video_id'] for r in read('EPIC_100_train.csv')}
    annotations = read('EPIC_100_validation.csv')
    classes = json.loads((ROOT/'weights/temporal/classes.json').read_text())
    model = TemporalRecognizer(ROOT/'weights/temporal/tsm_rgb.ckpt')
    scorer = Scorer(ROOT/'code/vocabulary.json',4)
    original = Scorer(ROOT/'code/vocabulary-baseline.json',4,shared=scorer)
    state = PipelineState(scorer,original,model)
    results = []
    for video_id in IDS:
        video = FOLDER/(video_id+'.mp4')
        assert digest(video) == record['files'][video_id]['sha256']
        started = time.perf_counter()
        duration, _ = probe(video)
        raw = frames(video,1)
        verbs,nouns = model.score_video(video,duration,4)
        truth = dict(actions=[dict(verb=classes['verb'][int(r['verb_class'])],noun=classes['noun'][int(r['noun_class'])],
                                  start=seconds(r['start_timestamp']),end=seconds(r['stop_timestamp']))
                             for r in annotations if r['video_id']==video_id])
        direct = [('tsm_original_backend',decode(verbs,nouns,classes,duration)),
                  ('joint_backend',model.recognize(video,duration,ROOT/'weights/temporal/classes.json',4,
                                                  decoder_path=ROOT/'weights/temporal_decoder.npz'))]
        for name,actions in direct:
            results.append(dict(video_id=video_id,profile=name,predictions=actions,metrics=metrics(actions,[],None,truth,duration)))
        for name,routing,decoder in [('old_gated','category',None),('scene_joint','scene',ROOT/'weights/temporal_decoder.npz')]:
            options = PipelineOptions('preserved',False,1,4,ROOT/'weights/temporal/tsm_rgb.ckpt',4,decoder,routing)
            output = infer_video(video,duration,raw,options,state)
            results.append(dict(video_id=video_id,profile=name,route=output.route,gate_margin=output.kitchen_margin,
                                predictions=output.actions,metrics=metrics(output.actions,output.highlights,output.label,truth,duration)))
        (FOLDER/'results.json').write_text(json.dumps(results,indent=2))
        print(video_id,'completed in',round(time.perf_counter()-started,2),'seconds',flush=True)
    summary = {name:float(np.mean([r['metrics'][f'action_f1_tiou_{t}'] for r in results if r['profile']==name for t in (.1,.3,.5)]))
               for name in ('tsm_original_backend','joint_backend','old_gated','scene_joint')}
    (FOLDER/'complete.json').write_text(json.dumps(dict(summary=summary,ground_truth_actions=sum(r['video_id'] in IDS for r in annotations)),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':main()
